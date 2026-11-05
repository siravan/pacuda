"""One mg7 [amplicol] run, as test_mg7_amplicol runs it, optionally on pacuda.
usage (from the MadGraph root): mg7_run.py RUN_DIR [--pacuda KERNEL] [--reuse] 'proc1' ['proc2' ...]
Generates the output (unless --reuse and it exists), applies the test's run-card edits,
retargets it to pacuda with KERNEL (pacuda.py retarget), runs generate_events and prints the
cross section. The output is kept."""
import glob, json, os, subprocess, sys, time

sys.path.insert(0, os.getcwd())
args = sys.argv[1:]
run_dir = os.path.abspath(args.pop(0))
kernel = None
reuse = False
while args and args[0].startswith("--"):
    flag = args.pop(0)
    if flag == "--pacuda":
        kernel = args.pop(0)
    elif flag == "--reuse":
        reuse = True
procs = args
t0 = time.time()
if reuse and not os.path.isdir(run_dir):
    sys.exit(f"mg7_run: --reuse: {run_dir} does not exist")
if not reuse:
    if not procs:
        sys.exit("mg7_run: no process to generate")
    os.makedirs(os.path.dirname(run_dir), exist_ok=True)
    import madgraph.interface.master_interface as MGCmd
    from tests.acceptance_tests.test_mg7_amplicol import _edit_run_card
    mg = MGCmd.MasterCmd()
    mg.no_notification()
    mg.exec_cmd("set automatic_html_opening False --no_save")
    for i, p in enumerate(procs):
        mg.exec_cmd(("generate " if i == 0 else "add process ") + p)
    mg.exec_cmd("output mg7 %s" % run_dir)
    _edit_run_card(os.path.join(run_dir, "Cards", "run_card.toml"))
t1 = time.time()
pacuda = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pacuda.py")
if kernel:
    subprocess.check_call([sys.executable, pacuda, "retarget", run_dir, "--kernel", kernel])
t2 = time.time()
with open(os.path.join(run_dir, "generate_events.log"), "w") as log:
    status = subprocess.call([sys.executable, os.path.join(run_dir, "bin", "generate_events"), "-f"],
                             cwd=run_dir, stdout=log, stderr=subprocess.STDOUT)
t3 = time.time()
infos = sorted(glob.glob(os.path.join(run_dir, "Events", "*", "info.json")))
r = json.load(open(infos[-1]))["process"] if infos else {}
print("%s pacuda=%s status=%d output=%.1fs retarget=%.1fs run=%.1fs mean=%r error=%r done=%r" % (
    os.path.basename(run_dir), kernel, status, t1 - t0, t2 - t1, t3 - t2, r.get("mean"), r.get("error"), r.get("done")))
