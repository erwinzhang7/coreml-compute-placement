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
import platform
import signal
import subprocess
import sys
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
    args = ap.parse_args()

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
    flush(sample_power=True)

    s = header["summary"]
    print()
    print(f"best {best:.1f} img/s, last {last:.1f} img/s, "
          f"sustained {s['sustained_fraction']:.3f} of peak "
          f"over {len(complete)} complete windows")
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
    main()
