# Invalid: M6 run of 2026-11-07, fan stuck off

Every file here was measured between about 11:30 and 13:30 EST on 2026-11-07. Throughout that
window the M6's fan was stuck in manual mode at 0 RPM, with the SoC around 104 C, because of a
bug in the fan-control app (fixed the same day). None of the harness's checks recorded fan state:
thermal pressure read nominal or moderate for most of the run, so it did not show the fault either.

Kept for the record only. The valid rerun is in `results/m6/`, written up in `m6/RESULTS.md`.
Every run there was gated by `m6/guarded.sh`, which records the fan before, during and after each
run and refuses to start or fails if the fan is off.
