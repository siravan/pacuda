#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""pacuda: run the LC matrix elements of an mg7 [amplicol] output on the recurrence runner.

  pacuda.py retarget OUTPUT [--kernel f64|df64|mixed|mixed64] [--batch N] [--sm SM ...] [--avx227-lanes L]
  pacuda.py undo OUTPUT

retarget, with no change to MadGraph:
  1. each subprocess's LC artifact (SubProcesses/<P>/lc) is converted into lc.pacuda
     (pacuda-convert; a subprocess it refuses keeps the stock library), its symjit kernels
     collected in OUTPUT/pacuda/kernels;
  2. the output's runner is generated and built with those kernels (pacuda-gen, runner/build.sh)
     into OUTPUT/lib/librecurrence.so;
  3. each converted subprocess's makefile builds the pacuda UMAMI library (umami/build.sh, linking
     that runner by $ORIGIN) in place of the stock one (the stock makefile is kept as
     makefile.stock), and is run (make): mg7 builds a subprocess's libraries only when they are
     missing (MadgraphProcess.compile_amplicol), so the stock ones would stay. The LC library is
     then checked to link librecurrence.so. The reweight (full-colour) library stays stock.
The model parameters mg7 will set (the param card's values) are set before recording.
generate_events then runs it as usual. undo restores the stock makefiles, rebuilds the stock
libraries (make) and removes what retarget added.

The kernel is the runner configuration the libraries use (default df64; f64 is RustiCol's
arithmetic). --avx227-lanes L: the symjit kernels in symjit 2.27.3's AVX order for whole blocks
of L points (bit-identity with a batched RustiCol; the stock one-point path needs the default).
--batch N: points per runner call (default 0: each process's RustiCol tile, e.g. 64 to 1024)."""
import argparse, glob, json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CONVERT = os.path.join(ROOT, "convert", "target", "release", "pacuda-convert")
GEN = os.path.join(ROOT, "gen", "target", "release", "pacuda-gen")
LC_RULE = "\t$(MAKE) -C lc/API/umami shared BUILD_DIR=$(BUILD)/lc RUSTICOL_CONFIG=$(RUSTICOL_CONFIG)"


def param_card_values(output):
    """The values mg7 sets on an amplicol library (subprocess_amplicol.amplicol_parameters): per
    mutable model parameter of subprocesses.json's `parameters` map, its (block, code) entry of
    Cards/param_card.dat, as a float (DECAY lines are the block "decay")."""
    blocks, block = {}, None
    for line in open(os.path.join(output, "Cards", "param_card.dat")):
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        head = words[0].lower()
        if head == "block":
            block = words[1].lower()
            blocks.setdefault(block, {})
        elif head == "decay":
            blocks.setdefault("decay", {})[(int(words[1]),)] = words[2]
            block = None
        elif block is not None:
            try:
                blocks[block][tuple(int(w) for w in words[:-1])] = words[-1]
            except ValueError:
                pass
    values = {}
    with open(os.path.join(output, "SubProcesses", "subprocesses.json")) as f:
        for s in json.load(f):
            for name, (blk, code) in s.get("parameters", {}).items():
                try:
                    values[name] = float(blocks[blk.lower()][tuple(code)])
                except (KeyError, ValueError):
                    continue
    return values


def linked_libraries(path):
    """The NEEDED entries of an ELF shared library (readelf -d)."""
    out = subprocess.run(["readelf", "-d", path], capture_output=True, text=True, check=True).stdout
    return [line.split("[", 1)[1].split("]", 1)[0] for line in out.splitlines() if "(NEEDED)" in line]


def me_paths(output, sub):
    """The LC libraries (me_path) of a subprocess directory, from subprocesses.json."""
    with open(os.path.join(output, "SubProcesses", "subprocesses.json")) as f:
        rel = os.path.relpath(sub, output)
        return sorted({os.path.join(output, s["me_path"]) for s in json.load(f) if s["path"] == rel})


def make(sub):
    result = run(["make"], cwd=sub, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"pacuda: make in {sub} failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}")


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
    parameters = os.path.join(work, "parameters.json")
    with open(parameters, "w") as f:
        json.dump(param_card_values(output), f)
    converted, refused = [], []
    for sub in subprocesses(output):
        makefile = os.path.join(sub, "makefile")
        if not open(makefile).read().count(LC_RULE) and not os.path.exists(makefile + ".stock"):
            sys.exit(f"pacuda: {makefile}: no stock LC rule to replace")
        result = run([CONVERT, "convert", os.path.join(sub, "lc"), "-o", os.path.join(sub, "lc.pacuda"),
                      "--kernels", kernels, "--kernel", args.kernel, "--batch", str(args.batch),
                      "--parameters", parameters],
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
        make(sub)
        for lib in me_paths(output, sub):
            if "librecurrence.so" not in linked_libraries(lib):
                sys.exit(f"pacuda: {lib} does not link librecurrence.so after make")
    print(f"pacuda: {len(converted)} subprocesses on the runner (kernel {args.kernel}, batch {args.batch or 'tile'})"
          + (f", {len(refused)} kept stock" if refused else ""))


def undo(args):
    output = os.path.abspath(args.output)
    for sub in subprocesses(output):
        stock = os.path.join(sub, "makefile.stock")
        if not os.path.exists(stock):
            continue
        shutil.move(stock, os.path.join(sub, "makefile"))
        for path in [os.path.join(sub, "lc.pacuda")]:
            if os.path.exists(path):
                os.remove(path)
        # the pacuda build of build/lc/libumami.so would look up to date to the stock sub-make
        shutil.rmtree(os.path.join(sub, "build", "lc"), ignore_errors=True)
        make(sub)
        for lib in me_paths(output, sub):
            if "librecurrence.so" in linked_libraries(lib):
                sys.exit(f"pacuda: {lib} still links librecurrence.so after make")
    for path in [os.path.join(output, "lib", "librecurrence.so")]:
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(os.path.join(output, "pacuda"), ignore_errors=True)
    print("pacuda: the stock makefiles and libraries are back")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("retarget")
    r.add_argument("output")
    r.add_argument("--kernel", default="df64", choices=["f64", "df64", "mixed", "mixed64"])
    r.add_argument("--batch", type=int, default=0)
    r.add_argument("--sm", type=int, nargs="*")
    r.add_argument("--avx227-lanes", type=int, default=0)
    u = sub.add_parser("undo")
    u.add_argument("output")
    args = parser.parse_args()
    {"retarget": retarget, "undo": undo}[args.command](args)


if __name__ == "__main__":
    main()
