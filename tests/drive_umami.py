"""Calls one UMAMI library as madspace's LC pass does (random helicity, helicity output), for a
profiler: CALLS batches of POINTS events of channel 0, random flavours, at alpha_s 0.118.
usage: drive_umami.py LIB.so ARTIFACT_DIR SUBPROCESSES_JSON [POINTS] [CALLS]"""
import ctypes as C, json, sys, time
import numpy as np

lib_path, artifact, subprocesses = sys.argv[1:4]
points = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
calls = int(sys.argv[5]) if len(sys.argv) > 5 else 5
s = json.load(open(subprocesses))[0]
n = len(s["flavors"][0]["options"][0])
rng = np.random.default_rng(5)


def rambo(n_out, ecm, k):
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


L = C.CDLL(lib_path)
L.umami_initialize.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
L.umami_matrix_element.argtypes = [C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t, C.c_size_t,
    C.POINTER(C.c_int), C.POINTER(C.c_void_p), C.c_size_t, C.POINTER(C.c_int), C.POINTER(C.c_void_p)]
h = C.c_void_p()
assert L.umami_initialize(C.byref(h), artifact.encode()) == 0
k = points
p = rambo(n - 2, 1000.0, k)
soa = np.ascontiguousarray(p.transpose(2, 1, 0).reshape(4 * n, k))
fl = rng.integers(0, len(s["flavors"]), k).astype(np.int32)
ch = np.full(k, s["umami_channel"], np.uint32)
alpha = np.full(k, 0.118)
me, col, hel = np.zeros(k), np.zeros(k, np.int32), np.zeros(k, np.int32)
ins = [(0, soa), (1, alpha), (2, fl), (4, rng.random(k)), (8, ch)]
outs = [(0, me), (2, col), (3, hel)]
ik = (C.c_int * len(ins))(*[a for a, _ in ins]); ip = (C.c_void_p * len(ins))(*[b.ctypes.data for _, b in ins])
ok = (C.c_int * len(outs))(*[a for a, _ in outs]); op = (C.c_void_p * len(outs))(*[b.ctypes.data for _, b in outs])
t = time.perf_counter()
for _ in range(calls):
    assert L.umami_matrix_element(h, k, k, 0, len(ins), ik, ip, len(outs), ok, op) == 0
dt = time.perf_counter() - t
print(f"{calls} calls of {k} events: {dt:.3f} s ({dt / calls / k * 1e6:.2f} us/event); sum ME {me.sum():.12g}")
