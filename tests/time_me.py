"""Times the UMAMI calls of an mg7 [amplicol] output: wraps each LC and ratio (reweight) library
in the timing shim tests/umami_timer.c, or unwraps them; prints the totals of a log.
usage: time_me.py wrap|unwrap|report OUTPUT
wrap: lib/X.so -> lib/X.real.so, the shim built at lib/X.so (it logs to OUTPUT/me_times.txt, one
line per library and process at exit). unwrap: the real libraries back. report: the totals per
library (LC, ratio) of OUTPUT/me_times.txt."""
import collections, json, os, shutil, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
command, output = sys.argv[1], os.path.abspath(sys.argv[2])
log = os.path.join(output, "me_times.txt")
libs = {}
with open(os.path.join(output, "SubProcesses", "subprocesses.json")) as f:
    for s in json.load(f):
        umami = os.path.join(output, s["path"], "lc", "API", "umami")
        libs[os.path.join(output, s["me_path"])] = ("lc", umami)
        libs[os.path.join(output, s["ratio_path"])] = ("ratio", umami)
if command == "wrap":
    if os.path.exists(log):
        os.remove(log)
    for lib, (name, umami) in libs.items():
        real = lib[:-3] + ".real.so"
        if os.path.exists(real):
            sys.exit(f"time_me: {real} exists (already wrapped)")
        shutil.move(lib, real)
        subprocess.check_call(["cc", "-O2", "-std=c11", "-fPIC", "-shared", f"-I{umami}",
                               f'-DREAL_LIB="{real}"', f'-DLOG_FILE="{log}"', f'-DNAME="{name}"',
                               "-o", lib, os.path.join(here, "umami_timer.c"), "-ldl"])
elif command == "unwrap":
    for lib in libs:
        real = lib[:-3] + ".real.so"
        if os.path.exists(real):
            shutil.move(real, lib)
elif command == "report":
    total = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
    for line in open(log):
        name, _, calls, events, me, init = line.split()
        t = total[name]
        t[0] += int(calls); t[1] += int(events); t[2] += float(me); t[3] += float(init)
    for name, (calls, events, me, init) in sorted(total.items()):
        per = me / events * 1e6 if events else 0.0
        print(f"{name}: {calls} calls, {events} events, {me:.3f} s in umami_matrix_element "
              f"({per:.2f} us/event), {init:.3f} s in umami_initialize")
else:
    sys.exit(__doc__)
