"""Phase 0 driver: one batch of UMAMI calls to an mg7 output's LC library, as madspace makes
them (momenta, alpha_s, flavour, random helicity, channel in; matrix element, colour and
helicity index out). Saves the outputs.
usage: drive.py OUTPUT_DIR OUT.npz SEED POINTS ALPHA_S [MODE]
MODE: sample (random helicity, helicity index out; the resolved path, default) or sum
(no helicity output; the selected path)"""
import ctypes as C, json, os, sys
import numpy as np

out_dir, out, seed, k, alpha_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
mode = sys.argv[6] if len(sys.argv) > 6 else "sample"
rng = np.random.default_rng(seed)


def rambo(n_out, ecm, n):
    q0 = -np.log(rng.random((n, n_out)) * rng.random((n, n_out)))
    c = 2 * rng.random((n, n_out)) - 1
    s = np.sqrt(1 - c * c)
    f = 2 * np.pi * rng.random((n, n_out))
    q = np.stack([q0, q0 * s * np.cos(f), q0 * s * np.sin(f), q0 * c], -1)
    Q = q.sum(1)
    M = np.sqrt(Q[:, 0] ** 2 - (Q[:, 1:] ** 2).sum(-1))
    b = -Q[:, 1:] / M[:, None]
    x = ecm / M
    g = Q[:, 0] / M
    a = 1 / (1 + g)
    bq = (b[:, None, :] * q[:, :, 1:]).sum(-1)
    p0 = x[:, None] * (g[:, None] * q[:, :, 0] + bq)
    pv = x[:, None, None] * (q[:, :, 1:] + b[:, None, :] * (q[:, :, 0:1] + a[:, None, None] * bq[:, :, None]))
    inc = np.zeros((n, 2, 4))
    inc[:, 0] = [ecm / 2, 0, 0, ecm / 2]
    inc[:, 1] = [ecm / 2, 0, 0, -ecm / 2]
    return np.concatenate([inc, np.concatenate([p0[..., None], pv], -1)], 1)


s = json.load(open(os.path.join(out_dir, "SubProcesses", "subprocesses.json")))[0]
L = C.CDLL(os.path.join(out_dir, s["me_path"]))
L.umami_initialize.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
L.umami_matrix_element.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
    C.POINTER(C.c_int), C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(C.c_int), C.POINTER(C.c_void_p)]
h = C.c_void_p()
assert L.umami_initialize(C.byref(h), os.path.join(out_dir, s["artifact"]).encode()) == 0
n = len(s["flavors"][0]["options"][0])
fl = rng.integers(0, len(s["flavors"]), k).astype(np.int32)
p = rambo(n - 2, 1000.0, k)
soa = np.ascontiguousarray(p.transpose(2, 1, 0).reshape(4 * n, k))
me, col, hel = np.zeros(k), np.zeros(k, np.int32), np.zeros(k, np.int32)
ins = [(0, soa), (1, np.full(k, alpha_s)), (2, fl), (8, np.full(k, s["umami_channel"], np.uint32))]
outs = [(0, me), (2, col)]
if mode == "sample":
    ins.append((4, rng.random(k)))
    outs.append((3, hel))
ik = (C.c_int * len(ins))(*[a for a, _ in ins]); ip = (C.c_void_p * len(ins))(*[b.ctypes.data for _, b in ins])
ok = (C.c_int * len(outs))(*[a for a, _ in outs]); op = (C.c_void_p * len(outs))(*[b.ctypes.data for _, b in outs])
assert L.umami_matrix_element(h, k, k, 0, len(ins), ik, ip, len(outs), ok, op) == 0
np.savez(out, me=me, color=col, helicity=hel, flavour=fl, momenta=p)
print(f"{os.path.basename(out_dir)}: {k} points, alpha_s {alpha_s}, {mode}: sum ME {me.sum():.10g}")
