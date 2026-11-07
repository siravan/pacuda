"""pac2json's legend of the source wavefunctions (convert/CONVENTIONS.md, Sources; transcribed
here) against the runner's source_values (pacman_recurrence_v1_test_sources), on random momenta of
every family, energy sign, helicity and chirality.
usage: check_sources.py librecurrence.so"""
import ctypes as C, math, random, sys
lib = C.CDLL(sys.argv[1])
D = C.POINTER(C.c_double)
sq2 = math.sqrt(2.0)
def sign(x): return math.copysign(1.0, x)
def ab_weyl(k, h, anti):
    E, px, py, pz = k
    if E > 0:
        s = 0.0 if (px == 0 and py == 0 and pz < 0) else math.sqrt(max(E + pz, 0))
        if anti: s = -s if s != 0 else 0.0
        b = -h * math.sqrt(2 * E) if s == 0 else ((-h * px / s + 1j * py / s) if anti else (h * px / s - 1j * py / s))
    else:
        s = 0.0 if (px == 0 and py == 0 and pz > 0) else math.sqrt(max(-(E + pz), 0))
        if not anti: s = -s if s != 0 else 0.0
        b = -h * math.sqrt(2 * abs(E)) if s == 0 else ((-h * px / s - 1j * py / s) if anti else (h * px / s + 1j * py / s))
    return complex(s), complex(b)
def w(name, k, h, ch, m):
    E, px, py, pz = k
    pt = math.hypot(px, py); ak = math.sqrt(px*px + py*py + pz*pz)
    if name == "scalar": return [1]
    if name.startswith("weyl"):
        anti = "anti" in name; a, b = ab_weyl(k, h, anti)
        if not anti:
            if E > 0: return [a, b] if (h, ch) == (1, 1) else [b, a] if (h, ch) == (-1, -1) else [0, 0]
            return [a, b] if (h, ch) == (-1, 1) else [b, a] if (h, ch) == (1, -1) else [0, 0]
        if E > 0: return [b, a] if (h, ch) == (1, 1) else [a, b] if (h, ch) == (-1, -1) else [0, 0]
        return [b, a] if (h, ch) == (-1, 1) else [a, b] if (h, ch) == (1, -1) else [0, 0]
    if name.startswith("dirac"):
        anti = "anti" in name
        if abs(m) < 1e-8:
            a, b = ab_weyl(k, h, anti)
            if not anti:
                cond = (h == 1) if E > 0 else (h == -1)
                return [a, b, 0, 0] if cond else [0, 0, b, a]
            cond = (h == -1) if E > 0 else (h == 1)
            return [0, 0, a, b] if cond else [b, a, 0, 0]
        n = sign(E) if E != 0 else -1.0
        n = (1 if E > 0 else -1) if not anti else (-1 if E > 0 else 1)
        nh = n * h
        om = [math.sqrt(abs(E) + ak), m / math.sqrt(abs(E) + ak)]
        f1 = (1 + n + (1 - n) * nh) / 2; f2 = (1 + n - (1 - n) * nh) / 2
        ip, im_ = int((1 + nh) // 2), int((1 - nh) // 2)
        g = [f1 * om[ip], f2 * om[im_]]
        sgnE = 1 if E > 0 else -1
        t = [sgnE * px, sgnE * py, sgnE * pz]
        r = max(ak + t[2], 0)
        c1 = 1.0 if ak == 0 else math.sqrt(r / (2 * ak))
        c2 = -nh if (r == 0 or ak == 0) else ((nh * t[0] + 1j * t[1]) if anti else (nh * t[0] - 1j * t[1])) / math.sqrt(2 * ak * r)
        chi = [c1, c2]
        if not anti: return [chi[im_] * g[1], chi[ip] * g[1], chi[im_] * g[0], chi[ip] * g[0]]
        return [chi[im_] * g[0], chi[ip] * g[0], chi[im_] * g[1], chi[ip] * g[1]]
    if name == "vector (massless)":
        if E > 0:
            if pt == 0: return [0, -h / sq2, 1j * sign(pz) / sq2, 0]
            return [0, (-h * px * pz / (E * pt) - 1j * py / pt) / sq2, (-h * py * pz / (E * pt) + 1j * px / pt) / sq2, h * pt / (E * sq2)]
        g_, P = -h, -E
        if pt == 0: return [0, -g_ / sq2, -1j * sign(pz) / sq2, 0]
        return [0, (-g_ * px * pz / (P * pt) + 1j * py / pt) / sq2, (-g_ * py * pz / (P * pt) - 1j * px / pt) / sq2, g_ * pt / (P * sq2)]
    # massive vector
    if E < 0: E, px, py, pz, h = -E, -px, -py, -pz, -h
    P = min(E, math.sqrt(px*px + py*py + pz*pz)); q = min(P, math.hypot(px, py)); l = 1 - abs(h); n = abs(h)
    if P == 0: return [0, -h / sq2, 1j * n / sq2, l]
    e = E / (m * P)
    if q == 0: mid = [-h / sq2, 1j * n * sign(pz) / sq2]
    else:
        z = pz * h / (P * q * sq2)
        mid = [l * px * e - px * z - 1j * n * py / (q * sq2), l * py * e - py * z + 1j * n * px / (q * sq2)]
    return [l * P / m, mid[0], mid[1], l * pz * e + h * q / (P * sq2)]
FAM = {"scalar": (0, 0), "weyl particle": (1, 0), "weyl antiparticle": (1, 1), "dirac particle": (2, 0), "dirac antiparticle": (2, 1),
       "vector (massless)": (3, 0), "vector (massive)": (3, 0)}
random.seed(3); cases = []
for name, (fam, ori) in FAM.items():
    for _ in range(300):
        m = 0.0 if name in ("vector (massless)",) or (name.startswith(("weyl", "scalar"))) else random.choice([0.0, 4.7, 91.188]) if name.startswith("dirac") else random.choice([80.4, 91.188])
        kind = random.random()
        if kind < 0.15: p3 = [0.0, 0.0, random.choice([-1, 1]) * random.uniform(1, 500)]
        elif kind < 0.2: p3 = [0.0, 0.0, 0.0]
        else: p3 = [random.uniform(-500, 500) for _ in range(3)]
        E = math.sqrt(sum(x * x for x in p3) + m * m)
        if E == 0: continue
        sgn = random.choice([1, -1])
        k = [sgn * E] + [sgn * x for x in p3]
        hs = [0] if fam == 0 else ([-1, 0, 1] if name == "vector (massive)" else [-1, 1])
        h = random.choice(hs); ch = random.choice([-1, 1]) if fam == 1 else 0
        code = fam | ori << 3 | (h + 2) << 5 | (ch + 1) << 8
        cases.append((name, k, h, ch, m, code))
n = len(cases)
codes = (C.c_uint32 * n)(*[c[5] for c in cases]); masses = (C.c_double * n)(*[c[4] for c in cases])
mom = (C.c_double * (4 * n))(*[x for c in cases for x in c[1]])
ore = (C.c_double * (4 * n))(); oim = (C.c_double * (4 * n))()
assert lib.pacman_recurrence_v1_test_sources(codes, masses, mom, n, ore, oim) == 0
worst = {}
for i, (name, k, h, ch, m, code) in enumerate(cases):
    want = [complex(ore[4 * i + j], oim[4 * i + j]) for j in range(4)]
    got = [complex(x) for x in w(name, k, h, ch, m)] + [0j] * 4
    scale = max(1e-300, max(abs(x) for x in want))
    err = max(abs(got[j] - want[j]) for j in range(4)) / scale
    if err > worst.get(name, (0, None))[0]: worst[name] = (err, (k, h, ch, m, want[:4], got[:4]))
for name in FAM:
    e, ex = worst.get(name, (0, None))
    print(f"{name:20s} worst relative difference {e:.1e}" + ("" if e < 1e-13 else f"  e.g. {ex}"))
