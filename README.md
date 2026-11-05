# pacuda

Runs pyAmpliCol's recurrence mode on NVIDIA GPUs inside MadGraph7, without RustiCol at run time.
A converter records the recurrence plan runs of a stock pyAmpliCol LC artifact into a `.pacuda`
file, and a C-only UMAMI library runs that file on the recurrence runner `librecurrence.so`.
MadGraph is not changed: `pacuda.py retarget` switches an mg7 output's subprocesses over.

Status: a full mg7 run of `g g > g g g` and `p p > j j j` gives the stock cross sections bit for
bit (f64 kernel). See `symjit-playground/plans/pacman-ptx/PACUDA.md` for the design and results.

## Parts

- `runner/`: the recurrence runner (`recurrence_runner.cu`, its C interface `pacman_recurrence.h`,
  the synthetic-plan test `rectest.c`). `[PACUDA_KERNELS=kernels.cuh] runner/build.sh OUT [SM...]`
  builds `librecurrence.so` (with an output's symjit kernels).
- `gen/`: `pacuda-gen`, CUDA device functions from pyAmpliCol's symjit plane kernels (symjit
  2.28.3); `check` compares them with symjit's CPU kernels.
- `rusticol/`: the patched RustiCol the converter builds on, vendored as patches:
  `UPSTREAM` (the MadGraph7 commit), `patches/` (rusticol-core only: the pacman runner hook and
  its record mode), `prepare.sh [MADGRAPH7_REPO]` (pyamplicol/ at that commit, from a local clone
  or GitHub, with the patches, into `build/rusticol`).
- `convert/`: `pacuda-convert` (`convert/build.sh [MADGRAPH7_REPO]`: prepares the RustiCol, then
  cargo builds; toolchain 1.99.1). `convert LC_DIR -o FILE.pacuda [--kernels DIR]`,
  `check FILE --runner librecurrence.so`, `record LC_DIR [--replay ...]`. Used only to convert.
- `umami/`: `umami_pacuda.c`, the UMAMI library (`umami/build.sh UMAMI_DIR RUNNER_DIR OUT.so`).
- `pacuda.py retarget OUTPUT [--kernel ...]` / `undo OUTPUT`: an mg7 output on pacuda, or back.
- `tests/`: `runner_variants.sh` (the runner's kernels), `compare_umami.py` (pacuda's UMAMI library
  against the stock one), `mg7_run.py` (an mg7 run as `test_mg7_amplicol` makes it),
  `symjit227/` (symjit 2.27.3's CPU outputs).
- `phase0/`: the recording runner and the trace experiments that showed the plan runs are a fixed
  recording.

Nothing reads environment variables at run time; the runner is configured from the `.pacuda` file.

License: 0BSD (as pyAmpliCol and the runner).
