# main.json conventions

The legend of the `main.json` that pac2json writes, word for word (until 2026-11-07 it was also in
`main.json` itself, as `configs.conventions` and `configs.instructions`). PacRunner (pac-julia) is
the reference implementation: its `src/wavefunctions.jl` (the sources) and `src/operations.jl` (the
instructions).

## Conventions

**complex:** a complex number is [re, im]

### Types

| Field | Type |
| --- | --- |
| `p<l>` | [f64; 4]: the momentum of external leg l, [E, px, py, pz] |
| `q<i>` | [f64; 4]: a momentum form, [E, px, py, pz] |
| `forms[i].terms` | Vec<(usize, f64)>: (leg, coefficient) |
| `crossing` | Option<Vec<{leg: usize, sign: f64}>>: per leg of the program, the caller's leg it reads and its sign |
| `c<i>` | [Complex<f64>; N], N = components: 1 scalar, 2 Weyl spinor, 4 vector or Dirac spinor, 6 antisymmetric tensor |
| `c<i>[j]` | Complex<f64> |
| `a<i>` | Complex<f64>: an amplitude |
| `s<i>` | Complex<f64>: a scale (a coupling or a mass; a mass and a kernel input use its real part, f64) |
| `x<i>` | Complex<f64>: an exact factor |
| `currents[i].initial` | "zero" \| "source" \| "written" \| "arena" |
| `currents[i].source` | Option<{wavefunction: String, family: "scalar" \| "weyl" \| "dirac" \| "vector", orientation: "particle" \| "antiparticle", leg: Option<usize>, momentum: q<i>, helicity: i32 (-1, 0, +1), chirality: i32 (-1, 0, +1), mass: s<i>, factor: x<i>}>: present exactly when initial is "source" |
| `bindings[i].inputs` | [f64; n]: the kernel's inputs, each a real number (the re or im part of a complex operand) |
| `bindings[i].outputs` | [usize; m]: the components of out that the kernel's m output pairs (Complex<f64>) go to |
| `reduction[h].terms` | Vec<(a<i>, f64)>: (amplitude, weight) |
| `outputs[h]` | f64: the value of helicity configuration h, \|M_h\|² with its weights |
| `output.group_factor` | (i64, i64): numerator, denominator |
| `output.helicities[h].legs` | [i32; legs]: the helicity of each leg |
| `inputs.alpha_s` | Option<{name: String, value: f64, power: i32}> |
| `examples: momenta` | [[f64; 4]; legs]: the caller's momenta of one point |
| `examples: outputs` | [f64; helicities] |
| `examples: total` | f64 |

**points:** every program is evaluated per phase-space point; all values below are per point

**names:** c<i>: a current (currents[i]); c<i>[j]: its component j (c<i> alone: component 0 or, as an operand, the whole current); a<i>: an amplitude (amplitudes[i]); q<i>: a momentum form (inputs.forms[i]); s<i>: a scale (inputs.scales[i]); x<i>: an exact factor (inputs.factors[i]); b<i>: a symjit binding (bindings[i]); k<i>: a symjit kernel (kernels[i]); p<l>: the momentum of external leg l

**momenta:** p<l> = sign × (the caller's momentum of leg inputs.crossing[l].leg), [E, px, py, pz]; with no crossing p<l> is the caller's leg l

**forms:** q<i> = Σ coefficient × p<leg> over its terms, per component (the first term added to 0.0, in term order)

**minkowski:** a·b = a0 b0 - a1 b1 - a2 b2 - a3 b3

**mode:** "=": the instruction's value is stored into its output; "+=": added to it (in body order: the sums are not reordered)

**initial:** a current's value before the body: "zero" (all components 0), "source" (the external wavefunction its `source` field describes), "written" (none: the body's first instruction writing it has mode "="), "arena" (read before any instruction writes it: whatever its planes hold)

### Sources

- **notation:** a source current's value is factor × w, w the wavefunction of its `wavefunction` field (RustiCol's wavefunctions.rs, in f64): k = (E, px, py, pz) its momentum form, h its helicity, ch its chirality, m the real part of its mass scale; p_T = √(px² + py²); |k| = √(px² + py² + pz²); sign(x) = ±1 with the sign of x (+1 for +0); an incoming leg has E < 0 (its form is minus its momentum)
- **scalar:** w = (1)
- **weyl particle:** E > 0: s = 0 if px = py = 0 and pz < 0, else √max(E + pz, 0); a = s; b = −h√(2E) if s = 0, else h·px/s − i·py/s; (h, ch) = (+1, +1): w = (a, b); (−1, −1): w = (b, a); else w = (0, 0). E ≤ 0: s = 0 if px = py = 0 and pz > 0, else −√max(−(E + pz), 0); a = s; b = −h√(2|E|) if s = 0, else h·px/s + i·py/s; (h, ch) = (−1, +1): w = (a, b); (+1, −1): w = (b, a); else w = (0, 0)
- **weyl antiparticle:** E > 0: s = 0 if px = py = 0 and pz < 0, else −√max(E + pz, 0); a = s; b = −h√(2E) if s = 0, else −h·px/s + i·py/s; (h, ch) = (+1, +1): w = (b, a); (−1, −1): w = (a, b); else w = (0, 0). E ≤ 0: s = 0 if px = py = 0 and pz > 0, else √max(−(E + pz), 0); a = s; b = −h√(2|E|) if s = 0, else −h·px/s − i·py/s; (h, ch) = (−1, +1): w = (b, a); (+1, −1): w = (a, b); else w = (0, 0)
- **dirac particle:** |m| < 1e-8: a, b as for a weyl particle of the same E branch; E > 0: h = +1: w = (a, b, 0, 0), else (0, 0, b, a); E ≤ 0: h = −1: w = (a, b, 0, 0), else (0, 0, b, a). Massive: n = sign(E), nh = n·h; ω = (√(|E| + |k|), m / √(|E| + |k|)); f1 = (1 + n + (1 − n)·nh)/2, f2 = (1 + n − (1 − n)·nh)/2; ip = (1 + nh)/2, im = (1 − nh)/2 (indices 0, 1); g = (f1·ω[ip], f2·ω[im]); t = n·(px, py, pz); r = max(|k| + t_z, 0); χ = (1 if |k| = 0, else √(r / (2|k|)); −nh if r = 0 or |k| = 0, else (nh·t_x − i·t_y) / √(2|k| r)); w = (χ[im]·g[1], χ[ip]·g[1], χ[im]·g[0], χ[ip]·g[0])
- **dirac antiparticle:** |m| < 1e-8: a, b as for a weyl antiparticle of the same E branch; E > 0: h = −1: w = (0, 0, a, b), else (b, a, 0, 0); E ≤ 0: h = +1: w = (0, 0, a, b), else (b, a, 0, 0). Massive: as a dirac particle with n = −sign(E), χ[1] = nh·t_x + i·t_y over the same denominator, and w = (χ[im]·g[0], χ[ip]·g[0], χ[im]·g[1], χ[ip]·g[1])
- **vector (massless):** m = 0 (gluon, photon). E > 0: w = (0, (−h·px·pz/(E·p_T) − i·py/p_T)/√2, (−h·py·pz/(E·p_T) + i·px/p_T)/√2, h·p_T/(E·√2)); for p_T = 0: w = (0, −h/√2, i·sign(pz)/√2, 0). E ≤ 0: with g = −h and P = −E: w = (0, (−g·px·pz/(P·p_T) + i·py/p_T)/√2, (−g·py·pz/(P·p_T) − i·px/p_T)/√2, g·p_T/(P·√2)); for p_T = 0: w = (0, −g/√2, −i·sign(pz)/√2, 0)
- **vector (massive):** m ≠ 0 (Z, W). If E < 0: k → −k, h → −h first. P = min(E, |k|), q = min(P, p_T), l = 1 − |h| (1 for the longitudinal state), n = |h|; P = 0: w = (0, −h/√2, i·n/√2, l); else, with e = E/(m·P) and z = pz·h/(P·q·√2): w = (l·P/m, l·px·e − px·z − i·n·py/(q·√2), l·py·e − py·z + i·n·px/(q·√2), l·pz·e + h·q/(P·√2)); for q = 0 the middle two are −h/√2 and i·n·sign(pz)/√2

**output:** outputs[h] = group_factor × Σ weight × |a|² over reduction[h]'s terms, in order (|a|² = fma(re, re, im × im)); the outputs are the helicity configurations of output.helicities; the matrix element is Σ_h outputs[h]

**alpha_s:** the program's scales and factors are those at inputs.alpha_s.value; outputs scale as (alpha_s / value)^power

## Instructions

**notation:** an instruction's operands are its fields: in[0], in[1] (input currents), out (output current), current (a current changed in place), momenta[0], momenta[1] and momentum (forms), amplitude; x[j] is component j of operand x (complex for currents, real for momenta: [E, px, py, pz]); j counts components from 0. In the formulas below l = in[0], r = in[1], o = the value computed, c = current, k = momentum, u = momenta[0], v = momenta[1]; i is the imaginary unit

### vertex

- **meaning:** out[j] (mode)= o[j] × coupling × factor for each component j of the type's output
- **weyl_vector_a:** l Weyl (2), r vector (4) → Weyl (2): o[0] = -i l[0] r[3] - i l[1] r[1] + l[1] r[2] + i l[0] r[0], o[1] = -l[0] r[2] - i l[0] r[1] + i l[1] r[0] + i l[1] r[3]
- **weyl_vector_b:** l Weyl (2), r vector (4) → Weyl (2): o[0] = -l[1] r[2] + i l[0] r[0] + i l[0] r[3] + i l[1] r[1], o[1] = -i l[1] r[3] + l[0] r[2] + i l[0] r[1] + i l[1] r[0]
- **tensor_vector:** l antisymmetric tensor (6), r vector (4) → vector (4): o[0] = l[0] r[1] + l[1] r[2] + l[2] r[3], o[1] = l[0] r[0] + l[3] r[2] + l[4] r[3], o[2] = l[1] r[0] - l[3] r[1] + l[5] r[3], o[3] = l[2] r[0] - l[4] r[1] - l[5] r[2]
- **vector_wedge:** l vector (4), r vector (4) → antisymmetric tensor (6): o = (l[0] r[1] - l[1] r[0], l[0] r[2] - l[2] r[0], l[0] r[3] - l[3] r[0], l[1] r[2] - l[2] r[1], l[1] r[3] - l[3] r[1], l[2] r[3] - l[3] r[2])
- **three_vector:** l vector (4), r vector (4) → vector (4), the colour-ordered three-gluon vertex: o[j] = (l·r)(u[j] - v[j]) + 2((l·v) r[j] - (r·u) l[j])
- **momenta:** read by three_vector only; for the other types they are the momenta the inputs carry

### propagator

- **meaning:** current[j] = o[j] × factor / (k·k) for each component j
- **weyl_a:** c Weyl (2): o[0] = -i c[0] k[3] - i c[1] k[1] + c[1] k[2] + i c[0] k[0], o[1] = -c[0] k[2] - i c[0] k[1] + i c[1] k[0] + i c[1] k[3]
- **weyl_b:** c Weyl (2): o[0] = -c[1] k[2] + i c[0] k[0] + i c[0] k[3] + i c[1] k[1], o[1] = -i c[1] k[3] + c[0] k[2] + i c[0] k[1] + i c[1] k[0]
- **vector:** c vector (4), Feynman gauge: o[j] = -i c[j]

**scale:** current[j] = factor × current[j], j < components

**clear:** current[j] = 0, j < components

**closure:** amplitude += Σ_j in[0][j] × in[1][j] × component_factors[j] × factor, j < components

**source:** current = factor × the external wavefunction (RustiCol's wavefunctions.rs) of the leg's momentum

**kernel:** the symjit kernel of the binding on the binding's inputs (in f64); its output pair n (re, im) × factor (mode)= out[binding.outputs[n]]; with binding.direct and factor exactly 1 the pair is stored as it is

**binding inputs:** re|im in[0][j], re|im in[1][j], re|im out[j]: a part of a component of the kernel row's operands; momenta[0][j], momenta[1][j]: a component of its momenta; s<i>: the real part of scale s<i>; re|im factor: of its factor
