# Not used: a CPU job shared the machine

The first M5 Max bandwidth and LLM runs (2026-11-08 01:47-01:51 EST) had another user job running beside
them: two CPU threads, niced, no GPU work. It started the same minute. Its effect shows mainly in the
CPU streaming-read sweep (105.6 GB/s at 18 threads, against 297.9 when the machine was to itself) and in
one STREAM copy repeat at 375 GB/s. The runs in the parent folder were repeated with the whole machine
reserved. Kept for the record only.
