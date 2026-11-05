"""Compares two UMAMI libraries of one mg7 subprocess on the same inputs, bit for bit: the stock
pyAmpliCol one (RustiCol on the CPU) and the pacuda one (the .pacuda file on the runner).
usage: compare_umami.py STOCK.so PACUDA.so ARTIFACT_DIR SUBPROCESSES_JSON [POINTS]
The modes: madspace's (random helicity, helicity output), a fixed helicity, the helicity sum,
and madspace's at another alpha_s. Run the stock library with PYAMPLICOL_UMAMI_POINTWISE=1 (its
original one-event path)."""
import ctypes as C, json, sys
import numpy as np

stock_path, pacuda_path, artifact, subprocesses = sys.argv[1:5]
points = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
s = json.load(open(subprocesses))[0]
n = len(s["flavors"][0]["options"][0])


def rambo(rng, n_out, ecm, k):
    q0 = -np.log(rng.random((k, n_out)) * rng.random((k, n_out)))
    c = 2 * rng.random((k, n_out)) - 1
    sn = np.sqrt(1 - c * c)
    f = 2 * np.pi * rng.random((k, n_out))
    q = np.stack([q0, q0 * sn * np.cos(f), q0 * sn * np.sin(f), q0 * c], -1)
    Q = q.sum(1)
    M = np.sqrt(Q[:, 0] ** 2 - (Q[:, 1:] ** 2).sum(-1))
    b = -Q[:, 1:] / M[:, None]
    x = ecm / M
    g = Q[:, 0] / M
    a = 1 / (1 + g)
    bq = (b[:, None, :] * q[:, :, 1:]).sum(-1)
    p0 = x[:, None] * (g[:, None] * q[:, :, 0] + bq)
    pv = x[:, None, None] * (q[:, :, 1:] + b[:, None, :] * (q[:, :, 0:1] + a[:, None, None] * bq[:, :, None]))
    inc = np.zeros((k, 2, 4))
    inc[:, 0] = [ecm / 2, 0, 0, ecm / 2]
    inc[:, 1] = [ecm / 2, 0, 0, -ecm / 2]
    return np.concatenate([inc, np.concatenate([p0[..., None], pv], -1)], 1)


def library(path):
    L = C.CDLL(path)
    L.umami_initialize.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
    L.umami_get_meta.argtypes = [C.c_int, C.c_void_p]
    L.umami_matrix_element.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
        C.POINTER(C.c_int), C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(C.c_int), C.POINTER(C.c_void_p)]
    L.umami_set_parameter.argtypes = [C.c_void_p, C.c_char_p, C.c_double, C.c_double]
    h = C.c_void_p()
    assert L.umami_initialize(C.byref(h), artifact.encode()) == 0, path
    return L, h


def call(lib, ins, outs, k):
    L, h = lib
    ik = (C.c_int * len(ins))(*[a for a, _ in ins]); ip = (C.c_void_p * len(ins))(*[b.ctypes.data for _, b in ins])
    ok = (C.c_int * len(outs))(*[a for a, _ in outs]); op = (C.c_void_p * len(outs))(*[b.ctypes.data for _, b in outs])
    status = L.umami_matrix_element(h, k, k, 0, len(ins), ik, ip, len(outs), ok, op)
    return status


libs = {"stock": library(stock_path), "pacuda": library(pacuda_path)}
nh = C.c_int(); libs["stock"][0].umami_get_meta(3, C.byref(nh))
rng = np.random.default_rng(7)
k = points
p = rambo(rng, n - 2, 1000.0, k)
soa = np.ascontiguousarray(p.transpose(2, 1, 0).reshape(4 * n, k))
fl = rng.integers(0, len(s["flavors"]), k).astype(np.int32)
ch = np.full(k, s["umami_channel"], np.uint32)
rh = rng.random(k)
hel_in = rng.integers(0, nh.value, k).astype(np.int32)
failed = 0
for mode in ["sample", "helicity", "sum", "sample alpha_s 0.13"]:
    alpha = np.full(k, 0.13 if "0.13" in mode else 0.118)
    res = {}
    for name, lib in libs.items():
        me, col, hel = np.zeros(k), np.zeros(k, np.int32), np.zeros(k, np.int32)
        ins = [(0, soa), (1, alpha), (2, fl), (8, ch)]
        outs = [(0, me), (2, col)]
        if mode.startswith("sample"):
            ins.append((4, rh))
            outs.append((3, hel))
        elif mode == "helicity":
            ins.append((6, hel_in))
        status = call(lib, ins, outs, k)
        res[name] = (status, me, col, hel)
    (s1, me1, c1, h1), (s2, me2, c2, h2) = res["stock"], res["pacuda"]
    if s1 != 0 or s2 != 0:
        print(f"{mode}: status stock {s1}, pacuda {s2}")
        failed += 1
        continue
    differ = int((me1.view(np.uint64) != me2.view(np.uint64)).sum())
    nz = me1 != 0
    rel = np.abs(me2[nz] / me1[nz] - 1).max() if nz.any() else 0.0
    print(f"{mode}: {differ} of {k} matrix elements differ in bits (max rel {rel:.1e}); "
          f"colours equal {np.array_equal(c1, c2)}, helicities equal {np.array_equal(h1, h2)}; sum stock {me1.sum():.12g}")
    if mode == "sample" and (differ or not np.array_equal(h1, h2) or not np.array_equal(c1, c2)):
        failed += 1
print("FAILED" if failed else "OK")
sys.exit(1 if failed else 0)
