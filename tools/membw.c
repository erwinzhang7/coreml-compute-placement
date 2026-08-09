// SPDX-License-Identifier: MIT
//
// Streaming memory bandwidth on Apple Silicon, measured per CPU core tier.
//
// The question this exists to answer: on a unified-memory Mac, can the CPU cores
// pull enough DRAM bandwidth to make CPU-side compute (KTransformers-style MoE
// expert offload) worthwhile, or does the GPU own the bus? Peak spec figures do
// not answer that -- the CPU cluster's achievable fraction is set by core count,
// outstanding-miss capacity and fabric ports, not by DRAM peak.
//
// Read-only streaming is the primary metric on purpose: weight-streaming during
// decode is read-dominated, so a read test models it better than a copy or triad.
//
// macOS gives no hard CPU affinity, so core tiers are steered with QoS classes,
// which the scheduler honours in practice:
//   --qos ui  -> QOS_CLASS_USER_INTERACTIVE, lands on the top tier
//   --qos bg  -> QOS_CLASS_BACKGROUND, lands on the lower tier
// Verify placement with `powermetrics --samplers cpu_power` while a run is going;
// QoS is a hint, not a guarantee.
//
// Build: cc -O3 -o membw membw.c
#define _DARWIN_C_SOURCE
#include <arm_neon.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// 128-byte cache lines on this hardware (hw.cachelinesize), so one loop
// iteration consumes exactly one line: 16 uint64 = 128 bytes.
#define ELEMS_PER_ITER 16

// CLOCK_MONOTONIC specifically, not mach_absolute_time: the contention harness
// needs to compare this program's timed window against one measured in Python,
// and time.clock_gettime(CLOCK_MONOTONIC) there is the same clock. CLOCK_UPTIME_RAW
// is not -- it excludes sleep, so on a laptop that has been up for days the two
// bases differ by hours and every overlap calculation silently becomes garbage.
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// Eight independent accumulators so the adds never serialise on a single
// dependency chain; the loop should be limited by outstanding loads, not ALU.
static uint64_t stream_read(const uint64_t *restrict p, size_t n) {
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
    uint64x2_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    size_t i = 0;
    for (; i + ELEMS_PER_ITER <= n; i += ELEMS_PER_ITER) {
        a0 = vaddq_u64(a0, vld1q_u64(p + i + 0));
        a1 = vaddq_u64(a1, vld1q_u64(p + i + 2));
        a2 = vaddq_u64(a2, vld1q_u64(p + i + 4));
        a3 = vaddq_u64(a3, vld1q_u64(p + i + 6));
        a4 = vaddq_u64(a4, vld1q_u64(p + i + 8));
        a5 = vaddq_u64(a5, vld1q_u64(p + i + 10));
        a6 = vaddq_u64(a6, vld1q_u64(p + i + 12));
        a7 = vaddq_u64(a7, vld1q_u64(p + i + 14));
    }
    uint64x2_t s = vaddq_u64(vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3)),
                             vaddq_u64(vaddq_u64(a4, a5), vaddq_u64(a6, a7)));
    uint64_t r = vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1);
    for (; i < n; i++) r += p[i];
    return r;
}

// macOS has no pthread_barrier_t, so here is the minimum viable one.
typedef struct {
    pthread_mutex_t m;
    pthread_cond_t c;
    int count, target, generation;
} barrier_t;

static void barrier_init(barrier_t *b, int target) {
    pthread_mutex_init(&b->m, NULL);
    pthread_cond_init(&b->c, NULL);
    b->count = 0;
    b->target = target;
    b->generation = 0;
}

static void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->m);
    int gen = b->generation;
    if (++b->count == b->target) {
        b->count = 0;
        b->generation++;
        pthread_cond_broadcast(&b->c);
    } else {
        while (gen == b->generation) pthread_cond_wait(&b->c, &b->m);
    }
    pthread_mutex_unlock(&b->m);
}

typedef struct {
    const uint64_t *base;
    size_t n;           // elements in this thread's slice
    double deadline;    // absolute stop time
    barrier_t *bar;
    uint64_t bytes;     // out: bytes this thread actually streamed
    uint64_t sink;      // out: keeps the reads from being optimised away
    double t0, t1;      // out: this thread's own window
} job_t;

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    // Touch the slice first so every page is faulted in and the timed region
    // measures DRAM traffic rather than the VM fault path.
    volatile uint64_t warm = stream_read(j->base, j->n);
    (void)warm;

    barrier_wait(j->bar);
    j->t0 = now_sec();
    uint64_t total = 0, sink = 0;
    while (now_sec() < j->deadline) {
        sink += stream_read(j->base, j->n);
        total += (uint64_t)j->n * sizeof(uint64_t);
    }
    j->t1 = now_sec();
    j->bytes = total;
    j->sink = sink;
    return NULL;
}

static qos_class_t parse_qos(const char *s) {
    if (!strcmp(s, "ui")) return QOS_CLASS_USER_INTERACTIVE;
    if (!strcmp(s, "user")) return QOS_CLASS_USER_INITIATED;
    if (!strcmp(s, "default")) return QOS_CLASS_DEFAULT;
    if (!strcmp(s, "util")) return QOS_CLASS_UTILITY;
    if (!strcmp(s, "bg")) return QOS_CLASS_BACKGROUND;
    fprintf(stderr, "unknown qos '%s'\n", s);
    exit(2);
}

int main(int argc, char **argv) {
    int threads = 8;
    double gb = 8.0, secs = 3.0;
    const char *qos_name = "ui";
    int quiet = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gb") && i + 1 < argc) gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--secs") && i + 1 < argc) secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--qos") && i + 1 < argc) qos_name = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else {
            fprintf(stderr,
                    "usage: %s [--threads N] [--gb N] [--secs N] [--qos ui|user|default|util|bg] [--quiet]\n",
                    argv[0]);
            return 2;
        }
    }
    if (threads < 1) threads = 1;

    size_t bytes = (size_t)(gb * (1ull << 30));
    bytes &= ~(size_t)(ELEMS_PER_ITER * sizeof(uint64_t) - 1);
    size_t nelem = bytes / sizeof(uint64_t);

    // MAP_ANON gives zero-filled pages; VM_FLAGS_SUPERPAGE is not requested since
    // the default 16K pages on this platform are already large.
    uint64_t *buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }

    // First-touch with a real pattern; all-zero pages can be handled specially.
    for (size_t i = 0; i < nelem; i += 2048) buf[i] = i + 1;
    memset(buf, 0x5a, bytes);

    qos_class_t qc = parse_qos(qos_name);
    barrier_t bar;
    barrier_init(&bar, threads + 1);

    pthread_t *tid = calloc(threads, sizeof(pthread_t));
    job_t *jobs = calloc(threads, sizeof(job_t));
    size_t slice = (nelem / threads) & ~(size_t)(ELEMS_PER_ITER - 1);

    double deadline = now_sec() + secs + 1.0;  // +1s covers the warm-up pass
    for (int i = 0; i < threads; i++) {
        jobs[i] = (job_t){ .base = buf + (size_t)i * slice, .n = slice,
                           .deadline = deadline, .bar = &bar };
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_set_qos_class_np(&attr, qc, 0);
        if (pthread_create(&tid[i], &attr, worker, &jobs[i]) != 0) {
            perror("pthread_create"); return 1;
        }
        pthread_attr_destroy(&attr);
    }

    barrier_wait(&bar);          // released once every worker has warmed up
    double t0 = now_sec();
    for (int i = 0; i < threads; i++) pthread_join(tid[i], NULL);
    double t1 = now_sec();

    uint64_t total = 0, sink = 0;
    for (int i = 0; i < threads; i++) { total += jobs[i].bytes; sink += jobs[i].sink; }
    double gbps = (double)total / (t1 - t0) / 1e9;

    if (quiet) {
        // threads qos gbps t0 t1 -- t0/t1 let a caller compute exactly how much of
        // this window overlapped some other engine's, instead of assuming.
        printf("%d %s %.1f %.6f %.6f\n", threads, qos_name, gbps, t0, t1);
    } else {
        printf("threads=%-3d qos=%-8s buffer=%.1f GiB/thread-slice=%.2f GiB  "
               "elapsed=%.2fs  read=%.1f GB  bandwidth=%.1f GB/s\n",
               threads, qos_name, gb, (double)slice * 8 / (1 << 30),
               t1 - t0, total / 1e9, gbps);
    }
    if (sink == 0x1234) fprintf(stderr, "");  // defeat dead-code elimination
    munmap(buf, bytes);
    free(tid); free(jobs);
    return 0;
}
