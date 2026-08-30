#!/bin/zsh
# FOURTH floor series, deliberately at 16:00 on BOTH inf boxes.
#
# WHY A FOURTH. Series 1, 2 and 3 cannot separate the two readings of the
# session effect, because series 3 ran CONTIGUOUSLY with series 2 -- "later in
# the session" and "later in the day" were the same variable, so the design could
# not distinguish them however many points it collected.
#
#   DRIFT        something moves monotonically over a session (thermal soak of
#                the room, a background task finishing, cumulative uptime).
#                A series starting FRESH at 16:00 should begin near a fresh
#                morning series, not near the end of a long one.
#   TIME OF DAY  something depends on the wall clock. A fresh 16:00 series should
#                begin DISPLACED from a fresh morning series.
#
# #178 measured the effect as a near-linear drift at ~0.004/hour. That is
# consistent with both readings; only a fresh start at a late hour separates them.
#
# BOTH BOXES, because #172 measured machines differing at 7.1 sigma. One box
# cannot tell a box effect from a time-of-day one.
#
# SAME TOOL as series 1-3. #176's concurrency refusal and .partial rename stay
# held back until this finishes -- a tool swap mid-study is precisely the
# confound this study exists to measure.
#
# SELF-STARTING AND SELF-GUARDING. It waits for 16:00 rather than depending on
# someone launching it at the right minute, and it REFUSES if a walk is still
# running on the box. The frozen tool does NOT have #175's concurrency refusal --
# that is one of the changes being held back -- so the guard has to live here.
# A soak measured beside another heavy job is not a floor, it is a load test.
# The pattern is supplied by the caller so this repo names no other project:
#   SOAK_REFUSE_PATTERN='myrunner (jobA|jobB)' tools/floor_rep4.sh
# It is REQUIRED. Unset, the script exits rather than running unguarded --
# a guard that silently fails open is worse than no guard.
#
# THE GUARD IS FAULT-INJECTED, because it runs unattended and a guard that has
# only been reasoned about is a liability. Three cases, on a box with a live walk:
#   positive  walk running          -> FIRES, and named the job it matched
#   negative  no matching process   -> silent
#   edge      only grep self-matches -> silent
# That third case is the one that bites: `ps | grep X` normally matches its own
# grep, and a naive guard fires on an empty box forever.
#
# ON REFUSAL IT EXITS AND DOES NOT RETRY. That is deliberate -- a late-afternoon
# fresh start is the measurement, and silently sliding it later would make the
# series something other than what was designed. It logs the refusal and the
# process list to ~/floor_rep4.log, so a failure is discoverable rather than
# silent. The residual risk is that a deadline watcher fails to kill its walk and
# the series simply does not run; that shows up in the log within one check.
LOG=~/floor_rep4.log
echo "$(date +%H:%M:%S) armed, waiting for 16:00" >> $LOG
while [ "$(date +%H%M)" -lt 1600 ]; do sleep 60; done

: "${SOAK_REFUSE_PATTERN:?refusing: set it to a ps pattern for jobs that must not overlap a floor measurement}"
if ps -eo command | grep -E "$SOAK_REFUSE_PATTERN" | grep -qv grep; then
  echo "$(date +%H:%M:%S) REFUSING: a walk is still running, this would measure load not floor" >> $LOG
  ps -eo command | grep -E "$SOAK_REFUSE_PATTERN" | grep -v grep >> $LOG
  exit 1
fi
echo "$(date +%H:%M:%S) box is clear, starting series 4" >> $LOG

cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}" || exit 1
# Last hyphen-separated component of the local hostname. Produces the same
# values as the previous hardcoded strip, without naming the machines' owner.
BOX=$(scutil --get LocalHostName | sed 's/.*-//')
echo "=== 3.6 drift, FOURTH series, fresh start at 16:00, box=$BOX  $(date +%H:%M:%S)"
for i in 01 02 03 04 05 06 07 08 09 10 11 12; do
  out="results/soak/rep4floor-${BOX}-siglip-CPU_AND_GPU-${i}.json"
  [ -f "$out" ] && { echo "SKIP $i"; continue; }
  echo "--- gpu $i  $(date +%H:%M:%S)"
  ./.venv/bin/python tools/thermal_soak.py siglip-vision-b16.mlpackage --units CPU_AND_GPU \
    --seconds 120 --window 10 --out "$out" 2>&1 | grep -E "^best|quiet|WARNING"
  sleep 90
done
echo "=== done $(date +%H:%M:%S)"
