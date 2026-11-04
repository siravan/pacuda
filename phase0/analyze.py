"""Phase 0 analysis of the recorded runner calls of one output (phase0/results/OUTPUT-*.jsonl).
usage: analyze.py OUTPUT"""
import collections, json, os, sys
import numpy as np

res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
o = sys.argv[1]
names = ["base", "momenta", "batch", "alphas", "sum"]
traces = {n: [json.loads(l) for l in open(f"{res}/{o}-{n}.jsonl")] for n in names}
power = json.load(open(os.path.join(os.path.dirname(res), "outputs", o, "SubProcesses", "subprocesses.json")))[0]["qcd_power"]
npz = {n: np.load(f"{res}/{o}-{n}.npz") for n in names + ["cpu", "cpusum"]}


def sig(call):
    return json.dumps([call["call"], call["static"]], sort_keys=True)


def summary(name, calls):
    runs = [c for c in calls if c["call"] == "run"]
    reds = [c for c in calls if c["call"] == "reduce"]
    keys = collections.Counter(c["static"]["key"] for c in runs)
    flags = collections.Counter(c["static"]["flags"] for c in runs)
    host_currents = sum(bool(c["static"]["upload_current"]) for c in runs)
    host_forms = sum(not (c["static"]["flags"] & 2) for c in runs)
    downloads = sum(bool(c["static"]["download_amplitude"]) and not (c["static"]["flags"] & 1) for c in runs)
    points = collections.Counter(c["data"]["points"] for c in runs)
    threads = len({c["tid"] for c in calls})
    bad = [c for c in calls if c["status"] != 0]
    print(f"  {name}: {len(runs)} runs, {len(reds)} reduces, {len(keys)} programs, {threads} threads, "
          f"flags {dict(flags)}, points/run {dict(sorted(points.items()))}, failed {len(bad)}")
    print(f"    host currents in {host_currents}, host forms in {host_forms}, "
          f"amplitudes copied back in {downloads} runs")
    return runs, reds


print(o)
for n in names:
    summary(n, traces[n])

# 1. the static part of the calls: same programs, tables and order?
base = traces["base"]
base_sigs = [sig(c) for c in base]
for n in names[1:]:
    sigs = [sig(c) for c in traces[n]]
    same_seq = sigs == base_sigs
    same_set = set(sigs) == set(base_sigs)
    print(f"  static vs base, {n}: same sequence {same_seq}, same set {same_set} "
          f"({len(set(sigs))} vs {len(set(base_sigs))} distinct)")
    if not same_set:
        diff = collections.Counter()
        bs = {json.dumps(c["static"], sort_keys=True) for c in base}
        for c in traces[n]:
            if json.dumps(c["static"], sort_keys=True) not in bs:
                b = next((x for x in base if x["call"] == c["call"] and
                          x["static"].get("key") == c["static"].get("key")), None)
                if b:
                    for f in c["static"]:
                        if c["static"][f] != b["static"].get(f):
                            diff[f] += 1
                else:
                    diff["(new program)" if c["call"] == "run" else "(new reduce)"] += 1
        print(f"    fields that differ: {dict(diff)}")

# 2. the data part: different momenta give different data, the same momenta the same
for n in ["sum", "momenta"]:
    a = [c["data"] for c in base if c["call"] == "run"]
    b = [c["data"] for c in traces[n] if c["call"] == "run"]
    if len(a) == len(b):
        same = sum(x == y for x, y in zip(a, b))
        print(f"  data vs base, {n}: {same}/{len(a)} runs with identical inputs and outputs")

# 3. the values
cpu = npz["cpu"]["me"]
for n in names:
    if n in ("momenta", "batch"):
        continue
    me = npz[n]["me"]
    cpu = npz["cpusum" if n == "sum" else "cpu"]["me"]
    ref = cpu if n != "alphas" else cpu * (0.13 / 0.118) ** power
    if n == "alphas":
        rel = np.abs(me / ref - 1).max()
        print(f"  {n}: ME(0.13) / (ME_cpu(0.118) (0.13/0.118)^{power}) - 1: max {rel:.1e}")
    else:
        print(f"  {n}: GPU == CPU bit for bit: {np.array_equal(me, cpu)}"
              + (f", helicities equal: {np.array_equal(npz[n]['helicity'], npz['cpu']['helicity'])}" if n == "base" else ""))

# 4. what the calls are made of (base)
for c in base:
    st = c["static"]
    if c["call"] == "run":
        print(f"  base run: {st['words']} words, planes {st['planes']}, {st['scales']} scales, "
              f"{st['factors']} factors, {st.get('externals')} externals, {st.get('terms')} form terms, "
              f"download {st['download_amplitude']}, zero {st.get('zero_amplitude')}")
