# pacuda

Runs pyAmpliCol's recurrence mode on NVIDIA GPUs inside MadGraph7, without RustiCol at run time:
a converter records the recurrence plan of a stock pyAmpliCol artifact into a `.pacuda` file,
and a C-only UMAMI library runs it with the recurrence runner `librecurrence.so`.

Status: phase 0 (the plan runs are a fixed recording) done; see
`symjit-playground/plans/pacman-ptx/PACUDA.md`.

- `runner/`: the recurrence runner (`recurrence_runner.cu`, its C interface `pacman_recurrence.h`,
  the synthetic-plan test `rectest.c`); `runner/build.sh [OUT] [SM...]` builds `librecurrence.so`.
- `tests/runner_variants.sh`: every kernel (f64, df64, mixed, mixed64) against f64.
- `phase0/`: the recording runner and the trace experiments on mg7 outputs.
- `convert/`, `umami/`: the converter and the C UMAMI wrapper (to come).

License: 0BSD (as pyAmpliCol and the runner).
