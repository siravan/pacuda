"""The kernels' shared libraries of a pac2json folder (built with sjb2obj) against symjit's CPU
kernels: check.bin of `pacuda-gen check KERNELS -d DIR` run on the folder's kernels in main.json's
order.
usage: check_kernel_libraries.py PAC2JSON_DIR check.bin"""
import ctypes as C, json, struct, sys
folder, checkbin = sys.argv[1], sys.argv[2]
main = json.load(open(folder + "/main.json"))
b = open(checkbin, "rb").read(); at = 0
def take(fmt, n=1):
    global at
    v = struct.unpack_from("<%d%s" % (n, fmt), b, at); at += struct.calcsize("<%d%s" % (n, fmt)); return v
nf, points = take("I", 2)
class Slice(C.Structure): _fields_ = [("ptr", C.POINTER(C.c_double)), ("len", C.c_size_t)]
assert nf == len(main["kernels"]), (nf, len(main["kernels"]))
for k in main["kernels"]:
    n_in, n_out = take("I", 2)
    inputs = take("d", n_in * points); scalar = take("d", n_out * points); take("d", n_out * points)
    lib = k["library"]; f = getattr(C.CDLL(folder + "/" + lib["path"]), lib["symbol"])
    f.restype = C.c_int
    assert (lib["states"], lib["outputs"]) == (n_in, n_out), (lib, n_in, n_out)
    ins = [(C.c_double * points)(*inputs[p * points:(p + 1) * points]) for p in range(n_in)]
    outs = [(C.c_double * points)() for _ in range(n_out)]
    planes = (Slice * (n_in + n_out))(*[Slice(C.cast(a, C.POINTER(C.c_double)), points) for a in ins + outs])
    for i in range(points):
        assert f(None, planes, i, None) == 0
    got = [outs[p][i] for p in range(n_out) for i in range(points)]
    differ = sum(struct.pack("<d", x) != struct.pack("<d", y) for x, y in zip(got, scalar))
    worst = max((abs(x - y) / max(abs(y), 1e-300) for x, y in zip(got, scalar)), default=0)
    print(f'{k["name"]} {k["sha256"][:12]}: {n_in} inputs, {n_out} outputs, {points} points: {differ} of {len(got)} differ in bits (max rel {worst:.1e})')
