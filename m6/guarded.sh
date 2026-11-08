#!/usr/bin/env bash
# Run one timed measurement with the machine's cooling verified around it.
#
#   m6/guarded.sh <name> <command...>
#
# The first M6 run was invalid because the fan sat at 0 RPM in manual mode for the whole
# session (a fan-control bug, SoC ~104 C), and nothing in the harness recorded it. So now:
#   1. refuse to start if a fan is stopped while the die is hot (see fan_ok below);
#   2. cool down: at least MIN_COOL s idle, then until the hottest die sensor is under
#      COOL_C (or MAX_COOL s pass, which is recorded rather than hidden);
#   3. record fan status + thermal pressure before and after, and sample the fan every 5 s
#      during the run, into out/thermal/<name>.*;
#   4. fail if, during the run, a fan sat at 0 RPM with a die over 85 C for >= 10 s.
# FAN is the fan-control CLI on this machine; its `status` prints JSON with fans[] and
# temperatures{} (SMC keys; Tp* CPU and Tg* GPU die sensors).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; shift
FAN=${FAN:-/usr/local/bin/fan}
MIN_COOL=${MIN_COOL:-60}; MAX_COOL=${MAX_COOL:-300}; COOL_C=${COOL_C:-60}
D=out/thermal; mkdir -p "$D"
PY=/usr/bin/python3   # stdlib only; works before any venv exists

snap() {  # one JSON line: wall time, fan, hottest Tp/Tg sensor, thermal pressure
    "$FAN" status 2>/dev/null | "$PY" -c '
import json, sys, time, ctypes, ctypes.util
d = json.load(sys.stdin)
t = {k: v for k, v in d["temperatures"].items() if k[:2] in ("Tp", "Tg")}
lib = ctypes.CDLL(ctypes.util.find_library("System")); tok = ctypes.c_int(0); st = ctypes.c_uint64(0)
lib.notify_register_check(b"com.apple.system.thermalpressurelevel", ctypes.byref(tok))
lib.notify_get_state(tok, ctypes.byref(st))
print(json.dumps({"t": round(time.time(), 1), "stage": sys.argv[1],
  "fans": [{"mode": f["mode"], "rpm": f["actual_rpm"], "target": f["target_rpm"]} for f in d["fans"]],
  "max_die_C": max(t.values()), "max_die_key": max(t, key=t.get),
  "thermal_pressure": int(st.value)}))' "$1"
}
# What a fault looks like. The fan-control app on this machine idles the fan in auto mode
# (0 RPM is normal on a cool die) and drives it in MANUAL mode with its own targets once the
# die warms, so manual mode alone is not a fault. The bug that invalidated the first run was a
# fan at 0 RPM while the die stayed hot. At the start: refuse if any fan is stopped on a die
# over 85 C. During the run: fail if that holds for two consecutive samples (>= 10 s), which
# spares the brief 0-RPM spin-up right after the app switches to manual.
fan_ok() { "$PY" -c '
import json, sys
d = json.loads(sys.argv[1])
sys.exit(1 if d["max_die_C"] > 85 and any(f["rpm"] == 0 for f in d["fans"]) else 0)' "$1"; }

S=$(snap preflight); echo "$S" | tee "$D/$NAME.pre.jsonl"
fan_ok "$S" || { echo "REFUSING: a fan is stopped while the die is over 85 C"; exit 3; }

sleep "$MIN_COOL"
t0=$(date +%s)
while :; do
    S=$(snap cooldown); echo "$S" >> "$D/$NAME.pre.jsonl"
    hot=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["max_die_C"])' "$S")
    "$PY" -c "import sys; sys.exit(0 if $hot < $COOL_C else 1)" && break
    (( $(date +%s) - t0 > MAX_COOL )) && { echo "cool-down timed out at ${hot} C (recorded)"; break; }
    sleep 10
done
S=$(snap start); echo "$S" | tee -a "$D/$NAME.pre.jsonl"
fan_ok "$S" || { echo "REFUSING: a fan is stopped while the die is over 85 C"; exit 3; }

( while :; do snap during >> "$D/$NAME.during.jsonl"; sleep 5; done ) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
set +e
"$@"
RC=$?
set -e
kill $MON 2>/dev/null || true; wait $MON 2>/dev/null || true
snap end | tee "$D/$NAME.post.jsonl"
if [[ -s "$D/$NAME.during.jsonl" ]]; then
    "$PY" - "$D/$NAME.during.jsonl" <<'EOF'
import json, os, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
stopped_hot = [r["max_die_C"] > 85 and any(f["rpm"] == 0 for f in r["fans"]) for r in rows]
bad = [rows[i] for i in range(1, len(rows)) if stopped_hot[i] and stopped_hot[i - 1]]
modes = sorted({f["mode"] for r in rows for f in r["fans"]})
rpm = [f["rpm"] for r in rows for f in r["fans"]]
print(f"during: n={len(rows)} rpm {min(rpm)}-{max(rpm)}  max die {max(r['max_die_C'] for r in rows):.1f} C  "
      f"max pressure {max(r['thermal_pressure'] for r in rows)}  modes {modes}  fault samples {len(bad)}")
sys.exit(4 if bad else 0)
EOF
fi
exit $RC
