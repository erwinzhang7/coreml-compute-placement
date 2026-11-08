// Per-engine ANE activity from IOReport, no root. libIOReport is private but needs no
// entitlement for reading; this is how macmon reads power. Read-only.
//
//   ioreport_ane list                  every channel whose group/subgroup/name mentions ANE
//   ioreport_ane sample <ms> <n>       n deltas of <ms> each, one JSON line per delta, for
//                                      those channels: simple counters as values, state
//                                      channels as per-state residency
//
// Build: clang -O2 -framework CoreFoundation -lIOReport m6/ioreport_ane.c -o bin/ioreport_ane
#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct IOReportSubscription *IOReportSubscriptionRef;
extern CFDictionaryRef IOReportCopyAllChannels(uint64_t, uint64_t);
extern CFStringRef IOReportChannelGetGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetSubGroup(CFDictionaryRef);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef);
extern int IOReportChannelGetFormat(CFDictionaryRef);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef,
                                                          CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t *);
extern int32_t IOReportStateGetCount(CFDictionaryRef);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int32_t);
extern int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);

static void cs(CFStringRef s, char *buf, size_t n) {
    buf[0] = 0;
    if (s) CFStringGetCString(s, buf, n, kCFStringEncodingUTF8);
}

static int matches(CFDictionaryRef ch) {
    char g[256], sg[256], nm[256];
    cs(IOReportChannelGetGroup(ch), g, sizeof g);
    cs(IOReportChannelGetSubGroup(ch), sg, sizeof sg);
    cs(IOReportChannelGetChannelName(ch), nm, sizeof nm);
    return strstr(g, "ANE") || strstr(sg, "ANE") || strstr(nm, "ANE") ||
           strstr(g, "ane") || strstr(sg, "ane") || strstr(nm, "ane");
}

int main(int argc, char **argv) {
    CFDictionaryRef all = IOReportCopyAllChannels(0, 0);
    CFArrayRef chans = CFDictionaryGetValue(all, CFSTR("IOReportChannels"));
    CFIndex n = CFArrayGetCount(chans);
    CFMutableArrayRef keep = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
    for (CFIndex i = 0; i < n; i++) {
        CFDictionaryRef ch = CFArrayGetValueAtIndex(chans, i);
        if (!matches(ch)) continue;
        CFArrayAppendValue(keep, ch);
        if (argc < 2 || strcmp(argv[1], "list") == 0) {
            char g[256], sg[256], nm[256], u[64];
            cs(IOReportChannelGetGroup(ch), g, sizeof g);
            cs(IOReportChannelGetSubGroup(ch), sg, sizeof sg);
            cs(IOReportChannelGetChannelName(ch), nm, sizeof nm);
            cs(IOReportChannelGetUnitLabel(ch), u, sizeof u);
            printf("%s | %s | %s | fmt=%d unit=%s\n", g, sg, nm, IOReportChannelGetFormat(ch), u);
        }
    }
    if (argc < 2 || strcmp(argv[1], "list") == 0) return 0;

    int ms = argc > 2 ? atoi(argv[2]) : 500, count = argc > 3 ? atoi(argv[3]) : 10;
    CFMutableDictionaryRef want = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks,
                                                            &kCFTypeDictionaryValueCallBacks);
    CFDictionarySetValue(want, CFSTR("IOReportChannels"), keep);
    CFMutableDictionaryRef subbed = NULL;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(NULL, want, &subbed, 0, NULL);
    if (!sub) { fprintf(stderr, "subscription failed\n"); return 1; }
    CFDictionaryRef prev = IOReportCreateSamples(sub, subbed, NULL);
    for (int k = 0; k < count; k++) {
        usleep(ms * 1000);
        CFDictionaryRef cur = IOReportCreateSamples(sub, subbed, NULL);
        CFDictionaryRef d = IOReportCreateSamplesDelta(prev, cur, NULL);
        CFRelease(prev);
        prev = cur;
        CFArrayRef items = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        printf("{\"t\":%.3f,\"ms\":%d,\"ch\":{", ts.tv_sec + ts.tv_nsec / 1e9, ms);
        for (CFIndex i = 0; items && i < CFArrayGetCount(items); i++) {
            CFDictionaryRef it = CFArrayGetValueAtIndex(items, i);
            char g[256], sg[256], nm[256];
            cs(IOReportChannelGetGroup(it), g, sizeof g);
            cs(IOReportChannelGetSubGroup(it), sg, sizeof sg);
            cs(IOReportChannelGetChannelName(it), nm, sizeof nm);
            printf("%s\"%s|%s|%s\":", i ? "," : "", g, sg, nm);
            int fmt = IOReportChannelGetFormat(it);
            if (fmt == 2) {  // state channel
                int32_t sc = IOReportStateGetCount(it);
                printf("{");
                for (int32_t s = 0; s < sc; s++) {
                    char sn[128];
                    cs(IOReportStateGetNameForIndex(it, s), sn, sizeof sn);
                    printf("%s\"%s\":%lld", s ? "," : "", sn, IOReportStateGetResidency(it, s));
                }
                printf("}");
            } else {
                printf("%lld", IOReportSimpleGetIntegerValue(it, NULL));
            }
        }
        printf("}}\n");
        fflush(stdout);
        CFRelease(d);
    }
    return 0;
}
