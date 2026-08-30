#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sustained-load soak: does a compute unit HOLD its throughput, or only reach it?

Every number in this repo so far is a burst measurement -- 30 iterations after a
warmup, over in a couple of seconds. That is the right way to compare peak rates
and the wrong way to decide what to ship. A unit that wins by 4.7x for two
seconds and then throttles to parity has not won.

The README lists "no sustained thermal soak" as a limitation. This closes it.

WHAT IT MEASURES. One model, one compute unit, hammered for `--seconds`, with
throughput bucketed into fixed windows. The headline is the ratio of the last
window to the best window: 1.00 means the unit held its rate, 0.60 means it gave
up 40%. Thermal pressure is sampled alongside so a decline can be attributed
rather than assumed.

THERMAL PRESSURE WITHOUT ROOT. powermetrics is the obvious instrument and it is
the wrong one here: it requires sudo, so anyone cloning this repo to check a
claim would hit a password prompt, and a benchmark nobody can run is not
evidence. macOS publishes the thermal pressure level on a notify(3) name that
any process can read, which is the same source NSProcessInfo.thermalState uses.
ctypes against libSystem reaches it with no extra dependency and no privilege.

    0 nominal   1 moderate   2 heavy   3 trapping   4 sleeping

A DECLINE IS NOT AUTOMATICALLY THROTTLING, and this is the trap the tool exists
to avoid walking into. Throughput can fall because the machine got hot, or
because something else started competing for the GPU, or because the display
woke. So the run records what it can distinguish them by: thermal level per
window, whether the machine is on AC or battery, and whether Low Power Mode is
on. On a laptop the battery/AC difference alone can dwarf the thermal effect,
and the M5 Max in this repo IS a laptop.

THERMAL PRESSURE IS NOT A TEMPERATURE, and reading it as one is the mistake this
paragraph exists to stop. `nominal` means macOS is not about to shed user work;
it does not mean the part is cool. The first soaks here reported nominal for
every window on a machine whose die was pinned at 97 C under a custom fan curve
-- the fans were holding the pressure signal down while the GPU clocked itself
down anyway. So a nominal reading RULES NOTHING OUT. It is only useful in the
other direction: a level above nominal is proof the machine was struggling.

To actually attribute a decline to heat you need the die temperature, and on
macOS that means `sudo powermetrics --samplers smc`. That is deliberately not
wired in here, because requiring root would make the soak unrunnable for anyone
cloning the repo. Run it alongside if you need the attribution.

FAN CURVE IS PART OF THE MACHINE. A custom fan curve changes sustained results
and nothing in the JSON can detect it, so record it in the filename or alongside
the result. Sustained numbers from an aggressively cooled laptop are not what a
stock one will do.

PARTIAL RESULTS SURVIVE A KILL. A soak is minutes long and the interesting ones
are the longest. Windows are appended to the output file as they complete, so
Ctrl-C costs the current window rather than the run. A window cut short by a
signal is marked `complete: false` and excluded from the headline, because an
earlier version appended it and let an arbitrarily short window become `last`.

ONLY FULL WINDOWS ARE SCHEDULED, and power is sampled only at the start and end
of a run. Both were audit findings. Clamping a final window to the deadline made
it shorter than the ones it is compared against, and calling pmset between every
window inserted a 12.5 ms unmeasured gap into what this tool calls a continuous
soak. Measured impact of both together on the M5 Max GPU figure: 0.837 to 0.847,
which is inside this measurement's own between-machine floor of 0.019.

    python tools/thermal_soak.py MODEL --units ANE --seconds 300
    python tools/thermal_soak.py MODEL --units CPU_AND_GPU --seconds 600 --window 5
"""
import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import coremltools as ct
import numpy as np

THERMAL_NAME = b"com.apple.system.thermalpressurelevel"
THERMAL_LEVELS = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping",
                  4: "sleeping"}


class Thermal:
    """Reads macOS thermal pressure via notify(3). No root, no pyobjc.

    Registers once and polls, rather than registering per sample: notify tokens
    are a process resource and re-registering in a hot loop leaks them.
    """

    def __init__(self):
        self.token = ctypes.c_int(0)
        self.ok = False
        try:
            self.lib = ctypes.CDLL(ctypes.util.find_library("System"))
            rc = self.lib.notify_register_check(ctypes.c_char_p(THERMAL_NAME),
                                                ctypes.byref(self.token))
            self.ok = (rc == 0)
        except OSError:
            self.lib = None

    def level(self):
        """Current level, or None if the name is unavailable on this OS."""
        if not self.ok:
            return None
        state = ctypes.c_uint64(0)
        if self.lib.notify_get_state(self.token, ctypes.byref(state)) != 0:
            return None
        return int(state.value)


def power_state():
    """AC vs battery and Low Power Mode.

    On a laptop these move sustained throughput more than temperature does, and
    neither is visible in a throughput curve. Recorded so a soak that declined
    on battery is not read as a thermal result.
    """
    out = {}
    ps = subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True).stdout
    out["source"] = ("AC" if "AC Power" in ps else
                     "battery" if "Battery Power" in ps else "unknown")
    for line in ps.splitlines():
        if "%" in line:
            out["battery"] = line.strip()
            break
    # macOS 26 lists lowpowermode in neither `pmset -g` nor `pmset -g custom`,
    # so absence is the OS not exposing it rather than a parse miss. Named
    # explicitly: a field reading "unknown" on every machine is indistinguishable
    # from one that is quietly broken.
    g = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    lpm = next((l for l in g.splitlines() if "lowpowermode" in l), "")
    out["lowpowermode"] = lpm.split()[-1] if lpm else "not-exposed-by-pmset"
    return out


class LoadSampler(threading.Thread):
    """Record what else was running, because a decline needs a suspect list.

    The docstring at the top of this file names three causes a throughput fall
    could have: heat, another process competing, or the display waking. The tool
    recorded evidence for the first and third and NOT the second, which is the
    one that has actually bitten this repo twice. The M5 Max ANE figures were
    taken on a contended box, came out 2x low, and became a published claim about
    silicon before they were re-run idle and retracted.

    It bit again on a `whisper` ANE soak that gave back 2.52% while every other
    ANE soak held to 0.14%. That run reached the SAME peak as two runs that did
    not decline, so it is not a contended peak, and the question of whether
    something started at t=70s cannot be answered from the file. It still cannot,
    for that run. It can be for every run after this.

    OFF THE MEASUREMENT THREAD, ON PURPOSE. Sampling inside the window loop is
    exactly the defect an audit already found here: calling pmset between windows
    put a measured 12.5 ms hole in what the tool calls a continuous soak. `ps`
    costs more than pmset, so this samples from its own thread and the predict
    loop never waits on it. The thread's own cost is real but constant across
    runs and off the critical path, which is the trade being made.

    What it keeps is deliberately small: the peak total non-self CPU seen, and
    the busiest few processes at that moment. Not a full time series, because the
    point is to answer "was this box quiet" and not to profile the machine.

    IT IS NOT FREE, AND THE COST IS MEASURED RATHER THAN ASSUMED. Three runs each
    way, alternating, mobilenet on the ANE:

        sampler on   3633.3  3636.5  3637.8   mean 3635.8
        sampler off  3641.4  3643.2  3643.0   mean 3642.5

    0.18% slower with it on, and the two groups do not overlap, so that is a real
    cost and not run-to-run noise. It would be dishonest to call it negligible
    when the headline ANE result is a 0.14% give-back.

    It does not touch that result, because `sustained_fraction` is last/best
    WITHIN one run and a constant offset cancels. The same six runs give 1.0000,
    1.0000, 0.9993 without and 1.0000, 1.0000, 0.9996 with. What does shift is the
    ABSOLUTE rate, which the soak README tells readers to compare across runs, so
    compare absolute rates only between runs with the same --load-every.
    """

    # `ps -o %cpu` on macOS is normalised so 100% is ONE core, not the machine.
    # The first cut of this class summed those numbers and called anything above
    # 25 "not quiet", which on a 14-core box is 25/1400 = 1.8% of capacity. It
    # fired on the first two soaks it ever ran, reporting 45.7% and 54.1% as
    # contention when the box was drawing about 3% of itself. Divide by the core
    # count before judging.
    #
    # It also counted the ssh session doing the checking, 15.2% of a core, so
    # looking at the box made it look busier. That is the pgrep-matches-itself
    # trap wearing a different hat, and it is why `quiet` is now a fraction of
    # machine capacity rather than a raw sum.
    def __init__(self, every=2.0, threshold=5.0, quiet_frac=0.15):
        super().__init__(daemon=True)
        self.every, self.threshold = every, threshold
        self.quiet_frac = quiet_frac
        self.ncpu = os.cpu_count() or 1
        self.stop = threading.Event()
        self.me = os.getpid()
        self.peak_cpu = 0.0
        # The busiest SINGLE process seen, kept separately from the sum.
        # The sum divided by core count is what hid a saturated core.
        self.peak_cpu_single = 0.0
        self.rival = None
        self.peak_at = None
        self.peak_procs = []
        self.samples = 0
        self.failed = None
        self.t0 = time.perf_counter()

    def run(self):
        while not self.stop.wait(self.every):
            try:
                out = subprocess.run(["ps", "-Ao", "pid=,pcpu=,comm="],
                                     capture_output=True, text=True, timeout=10).stdout
            except Exception as exc:                      # noqa: BLE001
                self.failed = str(exc)
                return
            busy, total = [], 0.0
            for line in out.splitlines():
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                try:
                    pid, cpu = int(parts[0]), float(parts[1])
                except ValueError:
                    continue
                # Exclude this process and its sampler. Counting ourselves would
                # report every soak as heavily contended, which is the same
                # self-matching error as `pgrep -f` finding its own command line.
                if pid == self.me:
                    continue
                total += cpu
                comm = parts[2].strip()[:60]
                if cpu >= self.threshold:
                    busy.append({"cpu": round(cpu, 1), "comm": comm})
                # Tracked on EVERY sample, not only at the peak of the total. A
                # rival can be running while the machine-wide sum is unremarkable
                # -- that is precisely the case the machine fraction misses -- so
                # looking only at busiest_at_peak would reintroduce the blind
                # spot one level down.
                if cpu > self.peak_cpu_single:
                    self.peak_cpu_single = cpu
                if cpu >= 50.0 and any(h in comm.lower() for h in self.RIVAL_HINTS):
                    if not self.rival or cpu > self.rival["cpu"]:
                        self.rival = {"comm": comm, "cpu": round(cpu, 1),
                                      "at_s": round(time.perf_counter() - self.t0, 1)}
            self.samples += 1
            if total > self.peak_cpu:
                self.peak_cpu = total
                self.peak_at = round(time.perf_counter() - self.t0, 1)
                self.peak_procs = sorted(busy, key=lambda d: -d["cpu"])[:5]

    def report(self):
        if self.failed:
            return {"readable": False, "why": self.failed}
        if not self.samples:
            return {"readable": False, "why": "no samples taken"}
        # Both scales, because only one of them is comparable across machines.
        # peak_other_cpu_pct is the raw ps sum, where 100 is one core.
        # peak_other_cpu_of_machine is that divided by the core count, which is
        # the number to threshold on and the number to compare between boxes.
        frac = self.peak_cpu / (100.0 * self.ncpu)
        return {
            "readable": True,
            "samples": self.samples,
            "cores": self.ncpu,
            "peak_other_cpu_pct": round(self.peak_cpu, 1),
            "peak_other_cpu_of_machine": round(frac, 4),
            "peak_at_s": self.peak_at,
            "busiest_at_peak": self.peak_procs,
            # TWO conditions, because the machine fraction alone has a blind spot
            # that cost real measurements. A concurrent soak driver is ONE
            # single-threaded Python process at ~100% of ONE core. On a 14-core
            # box that is 7.1% of the machine, so `frac` calls it quiet -- and a
            # second soak is the single condition these runs must not have. It
            # happened: a whisper GPU soak ran beside resnet50 ANE soaks, its own
            # record shows peak_other_cpu_pct 99.4, and the tool printed "box was
            # quiet". Four files had to be deleted.
            #
            # The single-process condition is NOT a bare CPU threshold. Sweeping
            # the corpus, 79 of 130 runs have some process above 50% of a core
            # and nearly all of them are WindowServer, PerfPowerServices,
            # airportd and friends -- ordinary background on any Mac. Rejecting
            # those would reject most of the corpus for nothing. What matters is
            # whether the competitor is doing COMPUTE, so the second condition
            # tests identity and load together.
            "quiet": frac < self.quiet_frac and not self._rival(),
            "peak_other_single_core_pct": round(self.peak_cpu_single, 1),
            "rival_at_peak": self._rival(),
        }

    # A process is a rival if it is working hard AND looks like a compute driver
    # rather than a system daemon. Named separately so the JSON records WHICH
    # process disqualified a run, not merely that one did.
    RIVAL_HINTS = ("python", "thermal_soak", "sweep.py", "coremltools",
                   "coverage_walk", "rolling_validate", "lightgbm", "mlx")

    def _rival(self):
        return self.rival


def machine():
    def s(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return {
        "cpu": s("sysctl", "-n", "machdep.cpu.brand_string"),
        "model": s("sysctl", "-n", "hw.model"),
        "macos": platform.mac_ver()[0],
        "coremltools": ct.__version__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--units", required=True,
                    choices=[u.name for u in ct.ComputeUnit])
    ap.add_argument("--seconds", type=float, default=300.0,
                    help="total soak length; the interesting ones are minutes")
    ap.add_argument("--window", type=float, default=10.0,
                    help="seconds per throughput bucket")
    ap.add_argument("--warmup", type=int, default=10,
                    help="calls before the clock starts, to force compilation "
                         "and first-touch allocation out of window 1")
    ap.add_argument("--out", default="")
    ap.add_argument("--load-every", type=float, default=2.0,
                    help="seconds between concurrent-load samples, 0 to disable. "
                         "Sampled from a separate thread so the predict loop "
                         "never waits on it")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="run even if another thermal_soak.py is active. Off by "
                         "default because two soaks on one box contend and the "
                         "load sampler cannot detect it")
    args = ap.parse_args()

    # REFUSE TO RUN BESIDE ANOTHER SOAK. This is the specific, decisive guard;
    # the load sampler is the diagnostic one and it cannot be relied on here,
    # because a second soak driver is a single-threaded process at ~100% of ONE
    # core, which is 7% of a fourteen-core machine and reads as quiet. That
    # happened, produced four unusable files, and the tool reported "box was
    # quiet" for every one of them.
    #
    # Checked by PID rather than by name matching alone, so this cannot find
    # itself -- the same trap the load sampler documents for `ps`.
    if not args.allow_concurrent:
        # Exclude by PROCESS GROUP, not by pid. Excluding only self and parent
        # still matched the shell that launched this run -- its command line
        # contains "thermal_soak.py" because it typed it -- so the guard fired
        # on an idle box. That is the pgrep-matches-itself trap this file warns
        # about in LoadSampler, walked into one function later.
        mypgid = os.getpgid(0)
        try:
            ps = subprocess.run(["ps", "-Ao", "pid=,pgid=,command="],
                                capture_output=True, text=True).stdout
        except OSError:
            ps = ""
        others = []
        for line in ps.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            pid, pgid, cmd = parts[0], parts[1], parts[2]
            if pgid.isdigit() and int(pgid) == mypgid:
                continue
            # A real soak is an INTERPRETER running the script, not a shell that
            # happens to name it.
            if "thermal_soak.py" not in cmd:
                continue
            head = cmd.split()[0].rsplit("/", 1)[-1].lower()
            if not head.startswith("python"):
                continue
            others.append(f"pid {pid}: {cmd.strip()[:70]}")
        if others:
            print("REFUSING: another soak is already running on this box.",
                  file=sys.stderr)
            for o in others:
                print("  " + o, file=sys.stderr)
            print("Two soaks on one machine contend, and the load sampler cannot "
                  "see it. Pass --allow-concurrent if that is genuinely what you "
                  "want to measure.", file=sys.stderr)
            return 2

    model = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit[args.units])
    spec = model.get_spec()
    name = spec.description.input[0].name
    shape = tuple(spec.description.input[0].type.multiArrayType.shape)
    batch = shape[0] if shape else 1
    x = {name: np.random.rand(*shape).astype(np.float32)}

    therm = Thermal()
    if not therm.ok:
        print("WARNING: thermal pressure unreadable; a decline cannot be "
              "attributed to heat on this run", file=sys.stderr)

    load = LoadSampler(every=args.load_every)
    if args.load_every > 0:
        load.start()

    for _ in range(args.warmup):
        model.predict(x)

    header = {
        "model": args.model, "units": args.units, "batch": batch,
        "seconds": args.seconds, "window": args.window,
        "machine": machine(), "power_start": power_state(),
        "thermal_readable": therm.ok,
    }
    windows = []

    stop = {"now": False}

    def onsig(signum, frame):
        stop["now"] = True
    signal.signal(signal.SIGINT, onsig)
    signal.signal(signal.SIGTERM, onsig)

    def flush(sample_power=False):
        # power_end goes INTO header, not into a copy of it. Writing it only to
        # the serialised dict left header["power_end"] permanently absent, so the
        # "power source changed" guard compared "battery" against None and fired
        # on every single run.
        #
        # sample_power is OFF for the per-window flush. power_state() shells out
        # to pmset twice, 12.5 ms measured, and doing that between every window
        # inserted an unmeasured idle gap into what this tool calls a continuous
        # soak, as well as eating into the deadline so the final window came up
        # short. An audit caught it. The gap was ~0.125% of a 10 s window and did
        # not move any published figure, but a soak with holes in it is not a soak.
        if sample_power:
            header["power_end"] = power_state()
            header["concurrent_load"] = load.report()
        if not args.out:
            return
        with open(args.out, "w") as fh:
            json.dump({**header, "windows": windows}, fh, indent=2)

    print(f"{args.units}  batch {batch}  soaking {args.seconds:.0f}s "
          f"in {args.window:.0f}s windows")
    print(f"{'t(s)':>6}  {'img/s':>9}  {'calls':>6}  {'thermal':>9}")

    t_start = time.perf_counter()
    deadline = t_start + args.seconds
    # Only schedule windows that can run to full length. A deadline-clamped final
    # window is shorter than the ones it is compared against, amortises the same
    # fixed per-window cost over fewer calls, and averages any drift over a
    # shorter horizon -- so it is not comparable to a full window even though its
    # rate is computed over its true elapsed time.
    while not stop["now"] and time.perf_counter() + args.window <= deadline:
        w_start = time.perf_counter()
        w_end = w_start + args.window
        calls = 0
        # Thermal is sampled at both ends of the window: a level that changes
        # mid-window would otherwise be attributed to whichever end was polled.
        lvl_start = therm.level()
        while time.perf_counter() < w_end and not stop["now"]:
            model.predict(x)
            calls += 1
        dt = time.perf_counter() - w_start
        lvl_end = therm.level()
        lvl = None if lvl_start is None else max(lvl_start, lvl_end)
        row = {
            "t": round(w_start - t_start, 3),
            "seconds": round(dt, 4),
            "calls": calls,
            "images_per_s": round(calls * batch / dt, 2) if dt > 0 else 0.0,
            "thermal": lvl,
            "thermal_name": THERMAL_LEVELS.get(lvl, "unreadable"),
        }
        # A window cut short by a signal is incomplete and must not become
        # `last`. The tool claimed a kill cost the current window; it did not,
        # it silently published it.
        row["complete"] = not stop["now"]
        windows.append(row)
        flush()
        print(f"{row['t']:>6.0f}  {row['images_per_s']:>9.1f}  {calls:>6}  "
              f"{row['thermal_name']:>9}")

    if not windows:
        sys.exit("no complete windows; raise --seconds or lower --window")

    complete = [w for w in windows if w.get("complete", True)]
    if len(complete) < 2:
        sys.exit(f"only {len(complete)} complete windows; raise --seconds or "
                 f"lower --window")
    rates = [w["images_per_s"] for w in complete]
    best, last = max(rates), rates[-1]
    # The starting thermal state is part of the result on a chip that throttles.
    # Six back-to-back soaks on an M5 Max gave GPU sustained 0.846, 0.847 and
    # 0.952, and the ordering is monotone in the PEAK: a box that starts cool
    # reaches a high peak and gives a lot back, a box already warm starts low and
    # gives little back, and SCORES BETTER on a statistic meant to mean "holds its
    # rate". So the fraction is a property of the unit AND the state it started
    # in. Record the peak so a reader can see which they are looking at, and
    # compare absolute last-window rates across runs rather than fractions.
    header["summary"] = {
        "windows": len(windows),
        "best_images_per_s": best,
        "last_images_per_s": last,
        # The number to quote. 1.00 means the unit held its peak for the whole
        # soak; anything lower is throughput the burst benchmarks do not see.
        "sustained_fraction": round(last / best, 4) if best else None,
        # Duration-weighted, not a mean of per-window rates: the windows are
        # equal length by construction now, but a mean of rates is the wrong
        # aggregate the moment they are not.
        "mean_images_per_s": round(
            sum(w["calls"] * batch for w in complete)
            / sum(w["seconds"] for w in complete), 2),
        "complete_windows": len(complete),
        "max_thermal": max((w["thermal"] for w in complete
                            if w["thermal"] is not None), default=None),
        "interrupted": stop["now"],
    }
    load.stop.set()
    flush(sample_power=True)

    s = header["summary"]
    print()
    print(f"best {best:.1f} img/s, last {last:.1f} img/s, "
          f"sustained {s['sustained_fraction']:.3f} of peak "
          f"over {len(complete)} complete windows")
    # Print it, do not just file it. The whole reason this field exists is that a
    # contended soak looked exactly like a clean one at the terminal, went into a
    # paper, and had to be retracted.
    cl = header.get("concurrent_load") or {}
    if cl.get("readable"):
        # Report the machine fraction, not the raw ps sum. `ps -o %cpu` counts one
        # core as 100, so on a 14-core box a raw 54 is under 4% of the machine and
        # saying "54%" invites reading it as half the hardware.
        pct = cl["peak_other_cpu_of_machine"] * 100
        if cl["quiet"]:
            print(f"box was quiet: peak other-process load {pct:.1f}% of "
                  f"{cl['cores']} cores over {cl['samples']} samples")
        else:
            busy = ", ".join(f"{p['comm']} {p['cpu']}%" for p in cl["busiest_at_peak"])
            print(f"WARNING: other work on this box. Peak other-process load "
                  f"{pct:.1f}% of {cl['cores']} cores at t={cl['peak_at_s']}s "
                  f"({busy}). A decline in this run may not be the compute unit.")
    else:
        print(f"concurrent load NOT recorded ({cl.get('why', 'unknown')}); a "
              f"decline in this run cannot be cleared of contention")
    # Warn when the run cannot have started cold. Without a reference this is a
    # heuristic, but a peak far below the best seen on this machine is the signal
    # that the box had not recovered from a previous soak.
    print()
    print("Peak this run: %.1f img/s. sustained_fraction is only comparable "
          "between runs that STARTED in the same thermal state; a warm start "
          "lowers the peak and RAISES the fraction. Compare last-window rates "
          "(%.1f img/s here) across runs, not fractions."
          % (best, last))
    if s["max_thermal"] in (0, None):
        print("thermal pressure stayed nominal. That does NOT rule heat out: "
              "pressure is macOS's signal that it is about to shed user work, "
              "not a die temperature. A machine held at 97 C by an aggressive "
              "fan curve reports nominal the whole time while its GPU clocks "
              "down. To attribute a decline, read the die temperature "
              "(sudo powermetrics --samplers smc), not this.")
    if header["power_start"]["source"] != header.get("power_end", {}).get("source"):
        print("POWER SOURCE CHANGED MID-RUN. Discard this soak.")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    # sys.exit, not a bare call: main() returns 2 when it refuses to run
    # beside another soak, and discarding that made the refusal exit 0.
    # A guard that declines to work and then reports success is worse
    # than no guard, because the chain believes it.
    sys.exit(main())
