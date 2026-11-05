#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""pacuda: run the LC matrix elements of an mg7 [amplicol] output on the recurrence runner.

  pacuda.py retarget OUTPUT [--kernel f64|df64|mixed|mixed64] [--sm SM ...] [--avx227-lanes L]
  pacuda.py undo OUTPUT

retarget, with no change to MadGraph:
  1. each subprocess's LC artifact (SubProcesses/<P>/lc) is converted into lc.pacuda
     (pacuda-convert; a subprocess it refuses keeps the stock library), its symjit kernels
     collected in OUTPUT/pacuda/kernels;
  2. the output's runner is generated and built with those kernels (pacuda-gen, runner/build.sh)
     into OUTPUT/lib/librecurrence.so;
  3. each converted subprocess's makefile builds the pacuda UMAMI library (umami/build.sh, linking
     that runner by $ORIGIN) in place of the stock one; the stock makefile is kept as
     makefile.stock. The reweight (full-colour) library stays stock.
generate_events then builds and runs it as usual. undo restores the stock makefiles and removes
what retarget added (the next generate_events relinks the stock libraries).

The kernel is the runner configuration the libraries use (default df64; f64 is RustiCol's
arithmetic). --avx227-lanes L: the symjit kernels in symjit 2.27.3's AVX order for whole blocks
of L points (bit-identity with a batched RustiCol; the stock one-point path needs the default)."""
import argparse, glob, json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT = os.path.join(ROOT, "convert", "target", "release", "pacuda-convert")
GEN = os.path.join(ROOT, "gen", "target", "release", "pacuda-gen")
LC_RULE = "\t$(MAKE) -C lc/API/umami shared BUILD_DIR=$(BUILD)/lc RUSTICOL_CONFIG=$(RUSTICOL_CONFIG)"


def subprocesses(output):
    with open(os.path.join(output, "SubProcesses", "subprocesses.json")) as f:
        paths = []
        for s in json.load(f):
            if s.get("backend") == "amplicol" and s["path"] not in paths:
                paths.append(s["path"])
    return [os.path.join(output, p) for p in paths]


def run(command, **kw):
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, **kw)


def retarget(args):
    output = os.path.abspath(args.output)
    work = os.path.join(output, "pacuda")
    kernels = os.path.join(work, "kernels")
    os.makedirs(kernels, exist_ok=True)
    converted, refused = [], []
    for sub in subprocesses(output):
        makefile = os.path.join(sub, "makefile")
        if not open(makefile).read().count(LC_RULE) and not os.path.exists(makefile + ".stock"):
            sys.exit(f"pacuda: {makefile}: no stock LC rule to replace")
        result = run([CONVERT, "convert", os.path.join(sub, "lc"), "-o", os.path.join(sub, "lc.pacuda"),
                      "--kernels", kernels, "--kernel", args.kernel],
                     capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            refused.append((sub, result.stderr.strip()))
            print(f"pacuda: {os.path.basename(sub)} keeps the stock library: {result.stderr.strip()}")
            continue
        converted.append(sub)
    if not converted:
        sys.exit("pacuda: no subprocess was converted")
    header = os.path.join(work, "kernels.cuh")
    gen = [GEN, "cuda", *sorted(glob.glob(os.path.join(kernels, "*.plane.symjit"))), "-o", header]
    if args.avx227_lanes:
        gen += ["--avx227-lanes", str(args.avx227_lanes)]
    if run(gen).returncode != 0:
        sys.exit("pacuda: pacuda-gen failed")
    env = dict(os.environ, PACUDA_KERNELS=header)
    if run([os.path.join(ROOT, "runner", "build.sh"), work, *map(str, args.sm or [])], env=env).returncode != 0:
        sys.exit("pacuda: the runner build failed")
    shutil.copy2(os.path.join(work, "librecurrence.so"), os.path.join(output, "lib", "librecurrence.so"))
    build = os.path.join(ROOT, "umami", "build.sh")
    for sub in converted:
        makefile = os.path.join(sub, "makefile")
        stock = makefile + ".stock"
        if not os.path.exists(stock):
            shutil.copy2(makefile, stock)
        text = open(stock).read()
        rule = (f"\tmkdir -p $(BUILD)/lc && {build} $(CURDIR)/lc/API/umami $(CURDIR)/../../lib "
                f"$(BUILD)/lc/libumami.so '$$ORIGIN'")
        open(makefile, "w").write(text.replace(LC_RULE, rule))
    print(f"pacuda: {len(converted)} subprocesses on the runner (kernel {args.kernel})"
          + (f", {len(refused)} kept stock" if refused else ""))


def undo(args):
    output = os.path.abspath(args.output)
    for sub in subprocesses(output):
        stock = os.path.join(sub, "makefile.stock")
        if os.path.exists(stock):
            shutil.move(stock, os.path.join(sub, "makefile"))
        for path in [os.path.join(sub, "lc.pacuda")]:
            if os.path.exists(path):
                os.remove(path)
    for path in [os.path.join(output, "lib", "librecurrence.so")]:
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(os.path.join(output, "pacuda"), ignore_errors=True)
    print("pacuda: the stock makefiles are back")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("retarget")
    r.add_argument("output")
    r.add_argument("--kernel", default="df64", choices=["f64", "df64", "mixed", "mixed64"])
    r.add_argument("--sm", type=int, nargs="*")
    r.add_argument("--avx227-lanes", type=int, default=0)
    u = sub.add_parser("undo")
    u.add_argument("output")
    args = parser.parse_args()
    {"retarget": retarget, "undo": undo}[args.command](args)


if __name__ == "__main__":
    main()
