// SPDX-License-Identifier: 0BSD
//! pac2json: a human-readable copy of a pyAmpliCol LC artifact's recurrence programs, recorded as
//! `pacuda-convert` records them (the patched RustiCol's record mode, no GPU needed).
//!
//!   pac2json LC_DIR -o OUT_DIR [--parameters FILE.json] [-n POINTS] [-e EXAMPLES]
//!   pac2json pack OUT_DIR -o FILE.pacuda [--kernel K] [--batch N]
//!
//! The first form writes OUT_DIR/main.json (the programs), OUT_DIR/examples.json (EXAMPLES
//! reference points per program, default 8: momenta and the CPU's outputs) and OUT_DIR/kernels/
//! (the symjit kernels the programs call, `<sha256>.plane.symjit`), then reads both JSON files back
//! and checks that they rebuild the recording (`pack`'s reconstruction). `--parameters` and `-n`:
//! as for `pacuda-convert convert`.
//! `pack`: the `.pacuda` file of a pac2json folder (for `pacuda-convert check` on a runner).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use pacuda_convert::format::{Colour, File, Process, Reduction, Template};
use pacuda_convert::program::{self, contribution_shape, Row, B_DESTINATION, B_FACTOR, B_MOMENTUM, B_PARENT, B_SCALE};
use pacuda_convert::record;
use serde_json::Value;

const FORMAT: u32 = 1;

// ---- an ordered JSON value and its printer -------------------------------------------------------

#[derive(Clone, Debug)]
enum J {
    O(Vec<(String, J)>),
    A(Vec<J>),
    F(f64),
    I(i64),
    S(String),
    B(bool),
    Null,
}

fn o(pairs: Vec<(&str, J)>) -> J {
    J::O(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}
fn s(v: impl Into<String>) -> J {
    J::S(v.into())
}
fn u(v: impl Into<i64>) -> J {
    J::I(v.into())
}
fn cx(re: f64, im: f64) -> J {
    J::A(vec![J::F(re), J::F(im)])
}
fn strs(v: &[String]) -> J {
    J::A(v.iter().map(|x| s(x.clone())).collect())
}

fn number(x: f64) -> String {
    if x.is_finite() { format!("{x:?}") } else { "null".into() }
}

fn quoted(text: &str) -> String {
    serde_json::to_string(text).unwrap()
}

/// one line, as short as JSON allows
fn compact(v: &J, out: &mut String) {
    match v {
        J::O(m) => {
            out.push('{');
            for (k, (key, x)) in m.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}: ", quoted(key));
                compact(x, out);
            }
            out.push('}');
        }
        J::A(a) => {
            out.push('[');
            for (k, x) in a.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                compact(x, out);
            }
            out.push(']');
        }
        J::F(x) => out.push_str(&number(*x)),
        J::I(x) => {
            let _ = write!(out, "{x}");
        }
        J::S(x) => out.push_str(&quoted(x)),
        J::B(x) => out.push_str(if *x { "true" } else { "false" }),
        J::Null => out.push_str("null"),
    }
}

/// a value on one line when it fits in `WIDTH` columns from its indentation, else one member per
/// line; an array of numbers wraps at `WIDTH`
fn pretty(v: &J, indent: usize, out: &mut String) {
    const WIDTH: usize = 160;
    let mut line = String::new();
    compact(v, &mut line);
    if indent + line.len() <= WIDTH || !matches!(v, J::O(_) | J::A(_)) {
        out.push_str(&line);
        return;
    }
    let pad = " ".repeat(indent + 2);
    match v {
        J::O(m) => {
            out.push_str("{\n");
            for (k, (key, x)) in m.iter().enumerate() {
                let _ = write!(out, "{pad}{}: ", quoted(key));
                pretty(x, indent + 2, out);
                out.push_str(if k + 1 < m.len() { ",\n" } else { "\n" });
            }
            let _ = write!(out, "{}}}", " ".repeat(indent));
        }
        J::A(a) if a.iter().all(|x| matches!(x, J::F(_) | J::I(_))) => {
            out.push_str("[\n");
            let mut row = String::new();
            for (k, x) in a.iter().enumerate() {
                let mut item = String::new();
                compact(x, &mut item);
                if k + 1 < a.len() {
                    item.push(',');
                }
                if !row.is_empty() && pad.len() + row.len() + 1 + item.len() > WIDTH {
                    let _ = writeln!(out, "{pad}{row}");
                    row.clear();
                }
                if !row.is_empty() {
                    row.push(' ');
                }
                row.push_str(&item);
            }
            let _ = writeln!(out, "{pad}{row}");
            let _ = write!(out, "{}]", " ".repeat(indent));
        }
        J::A(a) => {
            out.push_str("[\n");
            for (k, x) in a.iter().enumerate() {
                out.push_str(&pad);
                pretty(x, indent + 2, out);
                out.push_str(if k + 1 < a.len() { ",\n" } else { "\n" });
            }
            let _ = write!(out, "{}]", " ".repeat(indent));
        }
        _ => unreachable!(),
    }
}

fn write_json(path: &Path, v: &J) -> Result<()> {
    let mut text = String::new();
    pretty(v, 0, &mut text);
    text.push('\n');
    std::fs::write(path, text).with_context(|| path.display().to_string())
}

// ---- the legend --------------------------------------------------------------------------------

fn conventions() -> J {
    o(vec![
        ("complex", s("a complex number is [re, im]")),
        (
            "types",
            o(vec![
                ("p<l>", s("[f64; 4]: the momentum of external leg l, [E, px, py, pz]")),
                ("q<i>", s("[f64; 4]: a momentum form, [E, px, py, pz]")),
                ("forms[i].terms", s("Vec<(usize, f64)>: (leg, coefficient)")),
                ("crossing", s("Option<Vec<{leg: usize, sign: f64}>>: per leg of the program, the caller's leg it reads and its sign")),
                ("c<i>", s("[Complex<f64>; N], N = components: 1 scalar, 2 Weyl spinor, 4 vector or Dirac spinor, 6 antisymmetric tensor")),
                ("c<i>[j]", s("Complex<f64>")),
                ("a<i>", s("Complex<f64>: an amplitude")),
                ("s<i>", s("Complex<f64>: a scale (a coupling or a mass; a mass and a kernel input use its real part, f64)")),
                ("x<i>", s("Complex<f64>: an exact factor")),
                ("currents[i].initial", s("\"zero\" | \"source\" | \"written\" | \"arena\"")),
                (
                    "currents[i].source",
                    s("Option<{wavefunction: String, family: \"scalar\" | \"weyl\" | \"dirac\" | \"vector\", orientation: \"particle\" | \"antiparticle\", \
                       leg: Option<usize>, momentum: q<i>, helicity: i32 (-1, 0, +1), chirality: i32 (-1, 0, +1), mass: s<i>, factor: x<i>}>: \
                       present exactly when initial is \"source\""),
                ),
                ("bindings[i].inputs", s("[f64; n]: the kernel's inputs, each a real number (the re or im part of a complex operand)")),
                ("bindings[i].outputs", s("[usize; m]: the components of out that the kernel's m output pairs (Complex<f64>) go to")),
                ("reduction[h].terms", s("Vec<(a<i>, f64)>: (amplitude, weight)")),
                ("outputs[h]", s("f64: the value of helicity configuration h, |M_h|² with its weights")),
                ("output.group_factor", s("(i64, i64): numerator, denominator")),
                ("output.helicities[h].legs", s("[i32; legs]: the helicity of each leg")),
                ("inputs.alpha_s", s("Option<{name: String, value: f64, power: i32}>")),
                ("examples: momenta", s("[[f64; 4]; legs]: the caller's momenta of one point")),
                ("examples: outputs", s("[f64; helicities]")),
                ("examples: total", s("f64")),
            ]),
        ),
        ("points", s("every program is evaluated per phase-space point; all values below are per point")),
        (
            "names",
            s("c<i>: a current (currents[i]); c<i>[j]: its component j (c<i> alone: component 0 or, as an operand, the whole current); \
               a<i>: an amplitude (amplitudes[i]); q<i>: a momentum form (inputs.forms[i]); s<i>: a scale (inputs.scales[i]); \
               x<i>: an exact factor (inputs.factors[i]); b<i>: a symjit binding (bindings[i]); k<i>: a symjit kernel (kernels[i]); \
               p<l>: the momentum of external leg l"),
        ),
        (
            "momenta",
            s("p<l> = sign × (the caller's momentum of leg inputs.crossing[l].leg), [E, px, py, pz]; with no crossing p<l> is the caller's leg l"),
        ),
        ("forms", s("q<i> = Σ coefficient × p<leg> over its terms, per component (the first term added to 0.0, in term order)")),
        ("minkowski", s("a·b = a0 b0 - a1 b1 - a2 b2 - a3 b3")),
        (
            "mode",
            s("\"=\": the instruction's value is stored into its output; \"+=\": added to it (in body order: the sums are not reordered)"),
        ),
        (
            "initial",
            s("a current's value before the body: \"zero\" (all components 0), \"source\" (the external wavefunction its `source` field \
               describes), \"written\" (none: the body's first instruction writing it has mode \"=\"), \"arena\" (read before any \
               instruction writes it: whatever its planes hold)"),
        ),
        (
            "output",
            s("outputs[h] = group_factor × Σ weight × |a|² over reduction[h]'s terms, in order (|a|² = fma(re, re, im × im)); the outputs are the \
               helicity configurations of output.helicities; the matrix element is Σ_h outputs[h]"),
        ),
        (
            "alpha_s",
            s("the program's scales and factors are those at inputs.alpha_s.value; outputs scale as (alpha_s / value)^power"),
        ),
    ])
}

fn instructions() -> J {
    o(vec![
        (
            "notation",
            s("an instruction's operands are its fields: in[0], in[1] (input currents), out (output current), current (a current \
               changed in place), momenta[0], momenta[1] and momentum (forms), amplitude; x[j] is component j of operand x \
               (complex for currents, real for momenta: [E, px, py, pz]); j counts components from 0. In the formulas below \
               l = in[0], r = in[1], o = the value computed, c = current, k = momentum, u = momenta[0], v = momenta[1]; \
               i is the imaginary unit"),
        ),
        (
            "vertex",
            o(vec![
                ("meaning", s("out[j] (mode)= o[j] × coupling × factor for each component j of the type's output")),
                ("weyl_vector_a", s("l Weyl (2), r vector (4) → Weyl (2): o[0] = -i l[0] r[3] - i l[1] r[1] + l[1] r[2] + i l[0] r[0], o[1] = -l[0] r[2] - i l[0] r[1] + i l[1] r[0] + i l[1] r[3]")),
                ("weyl_vector_b", s("l Weyl (2), r vector (4) → Weyl (2): o[0] = -l[1] r[2] + i l[0] r[0] + i l[0] r[3] + i l[1] r[1], o[1] = -i l[1] r[3] + l[0] r[2] + i l[0] r[1] + i l[1] r[0]")),
                ("tensor_vector", s("l antisymmetric tensor (6), r vector (4) → vector (4): o[0] = l[0] r[1] + l[1] r[2] + l[2] r[3], o[1] = l[0] r[0] + l[3] r[2] + l[4] r[3], o[2] = l[1] r[0] - l[3] r[1] + l[5] r[3], o[3] = l[2] r[0] - l[4] r[1] - l[5] r[2]")),
                ("vector_wedge", s("l vector (4), r vector (4) → antisymmetric tensor (6): o = (l[0] r[1] - l[1] r[0], l[0] r[2] - l[2] r[0], l[0] r[3] - l[3] r[0], l[1] r[2] - l[2] r[1], l[1] r[3] - l[3] r[1], l[2] r[3] - l[3] r[2])")),
                ("three_vector", s("l vector (4), r vector (4) → vector (4), the colour-ordered three-gluon vertex: o[j] = (l·r)(u[j] - v[j]) + 2((l·v) r[j] - (r·u) l[j])")),
                ("momenta", s("read by three_vector only; for the other types they are the momenta the inputs carry")),
            ]),
        ),
        (
            "propagator",
            o(vec![
                ("meaning", s("current[j] = o[j] × factor / (k·k) for each component j")),
                ("weyl_a", s("c Weyl (2): o[0] = -i c[0] k[3] - i c[1] k[1] + c[1] k[2] + i c[0] k[0], o[1] = -c[0] k[2] - i c[0] k[1] + i c[1] k[0] + i c[1] k[3]")),
                ("weyl_b", s("c Weyl (2): o[0] = -c[1] k[2] + i c[0] k[0] + i c[0] k[3] + i c[1] k[1], o[1] = -i c[1] k[3] + c[0] k[2] + i c[0] k[1] + i c[1] k[0]")),
                ("vector", s("c vector (4), Feynman gauge: o[j] = -i c[j]")),
            ]),
        ),
        ("scale", s("current[j] = factor × current[j], j < components")),
        ("clear", s("current[j] = 0, j < components")),
        ("closure", s("amplitude += Σ_j in[0][j] × in[1][j] × component_factors[j] × factor, j < components")),
        ("source", s("current = factor × the external wavefunction (RustiCol's wavefunctions.rs) of the leg's momentum")),
        (
            "kernel",
            s("the symjit kernel of the binding on the binding's inputs (in f64); its output pair n (re, im) × factor (mode)= \
               out[binding.outputs[n]]; with binding.direct and factor exactly 1 the pair is stored as it is"),
        ),
        (
            "binding inputs",
            s("re|im in[0][j], re|im in[1][j], re|im out[j]: a part of a component of the kernel row's operands; momenta[0][j], \
               momenta[1][j]: a component of its momenta; s<i>: the real part of scale s<i>; re|im factor: of its factor"),
        ),
    ])
}

// ---- main.json from the recording ------------------------------------------------------------------

const FAMILIES: [&str; 4] = ["scalar", "weyl", "dirac", "vector"];

fn source_components(code: u32) -> u32 {
    match code & 7 {
        0 => 1,
        1 => 2,
        _ => 4,
    }
}

fn contribution_type(op: u32) -> &'static str {
    ["", "", "weyl_vector_a", "weyl_vector_b", "tensor_vector", "vector_wedge", "three_vector"][op as usize]
}
fn propagator_type(op: u32) -> &'static str {
    ["weyl_a", "weyl_b", "vector"][(op - 7) as usize]
}

/// How a current gets its value before the body uses it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Init {
    /// a clear (op 1) of its planes
    Zero,
    /// a source (op 12): row index
    Source(usize),
    /// its first writer has mode "="
    Written,
    /// read before any row writes it: the arena's content
    Arena,
}

struct Current {
    base: u32,
    count: u32,
    init: Init,
    /// the momentum form a source or a propagator names for it
    momentum: Option<u32>,
    read: bool,
}

/// The program's currents as values: the arena reuses planes, so a current is a range of planes
/// from the row that defines it (a source, a clear, a write with mode "=") to its next definition.
struct Analysis {
    currents: Vec<Current>,
    /// per row, the (current, component) of each plane operand that the row uses, in the order
    /// p0, p1, dst (sources: dst; clears, propagators, identities: base); None for an unused one
    operands: Vec<Vec<Option<(usize, u32)>>>,
    /// the rows that are a current's initial value (its source or clear)
    initializer: Vec<bool>,
}

/// (parent 0, parent 1, destination read, destination written) components of a symjit row
fn symjit_extents(inputs: &[u32], outputs: &[u32]) -> (u32, u32, u32, u32) {
    let (mut e0, mut e1, mut ed) = (0u32, 0u32, 0u32);
    for c in inputs {
        let (kind, which, n) = (c & 7, (c >> 3) & 1, c >> 5);
        match (kind, which) {
            (B_PARENT, 0) => e0 = e0.max(n + 1),
            (B_PARENT, _) => e1 = e1.max(n + 1),
            (B_DESTINATION, _) => ed = ed.max(n + 1),
            _ => {}
        }
    }
    (e0, e1, ed, outputs.iter().map(|c| c + 1).max().unwrap_or(0))
}

impl Analysis {
    fn of(rows: &[Row]) -> Result<Analysis> {
        let mut an = Analysis { currents: Vec::new(), operands: Vec::new(), initializer: Vec::new() };
        let mut live: BTreeMap<u32, usize> = BTreeMap::new();
        let mut binding: Option<(Vec<u32>, Vec<u32>)> = None;
        for (n, row) in rows.iter().enumerate() {
            let mut ops: Vec<Option<(usize, u32)>> = Vec::new();
            let mut initializer = false;
            match row {
                Row::Clear { base, count } => {
                    let (c, new) = an.define(&mut live, *base, *count, Init::Zero);
                    initializer = new;
                    ops.push(Some((c, 0)));
                }
                Row::Source { dst, code, form, .. } => {
                    let (c, new) = an.define(&mut live, *dst, source_components(*code), Init::Source(n));
                    if !new {
                        bail!("row {n}: a source into a current already defined");
                    }
                    an.currents[c].momentum = Some(*form);
                    initializer = true;
                    ops.push(Some((c, 0)));
                }
                Row::Contribution { op, init, p0, p1, dst, .. } => {
                    let (c0, c1, cd) = contribution_shape(*op);
                    ops.push(Some(an.read(&mut live, *p0, c0, n)?));
                    ops.push(Some(an.read(&mut live, *p1, c1, n)?));
                    ops.push(Some(an.write(&mut live, *dst, cd, *init, n)?));
                }
                Row::Propagator { op, base, form, .. } => {
                    let r = an.read(&mut live, *base, if *op == 9 { 4 } else { 2 }, n)?;
                    an.currents[r.0].momentum.get_or_insert(*form);
                    ops.push(Some(r));
                }
                Row::Identity { base, count, .. } => ops.push(Some(an.read(&mut live, *base, *count, n)?)),
                Row::Closure { p0, p1, count, .. } => {
                    ops.push(Some(an.read(&mut live, *p0, *count, n)?));
                    ops.push(Some(an.read(&mut live, *p1, *count, n)?));
                }
                Row::Binding { inputs, outputs, .. } => binding = Some((inputs.clone(), outputs.clone())),
                Row::Symjit { init, p0, p1, dst, .. } => {
                    let Some((inputs, outputs)) = &binding else { bail!("row {n}: a symjit row before any binding") };
                    let (e0, e1, ed, ew) = symjit_extents(inputs, outputs);
                    ops.push(if e0 > 0 { Some(an.read(&mut live, *p0, e0, n)?) } else { None });
                    ops.push(if e1 > 0 { Some(an.read(&mut live, *p1, e1, n)?) } else { None });
                    if ed > 0 && *init {
                        bail!("row {n}: a symjit row with mode \"=\" reads its destination");
                    }
                    ops.push(Some(if *init || ed == 0 { an.write(&mut live, *dst, ew.max(ed), *init, n)? } else { an.read(&mut live, *dst, ew.max(ed), n)? }));
                }
            }
            an.operands.push(ops);
            an.initializer.push(initializer);
        }
        Ok(an)
    }

    /// A new current of planes `base .. base + count`, or the live one of exactly these planes
    /// when nothing has read it since its definition (a clear then a write with mode "=").
    fn define(&mut self, live: &mut BTreeMap<u32, usize>, base: u32, count: u32, init: Init) -> (usize, bool) {
        let same = live.get(&base).copied().filter(|&c| {
            let cur = &self.currents[c];
            cur.base == base && cur.count == count && !cur.read && (base..base + count).all(|p| live.get(&p) == Some(&c))
        });
        if let Some(c) = same {
            return (c, false);
        }
        self.currents.push(Current { base, count, init, momentum: None, read: false });
        let c = self.currents.len() - 1;
        for p in base..base + count {
            live.insert(p, c);
        }
        (c, true)
    }

    /// the live current that holds planes `plane .. plane + count`, and the component of `plane`
    fn read(&mut self, live: &mut BTreeMap<u32, usize>, plane: u32, count: u32, row: usize) -> Result<(usize, u32)> {
        let c = match live.get(&plane) {
            Some(&c) => c,
            None => self.define(live, plane, count, Init::Arena).0,
        };
        let cur = &self.currents[c];
        if !(plane..plane + count).all(|p| live.get(&p) == Some(&c)) {
            bail!("row {row}: planes {plane}..{} are not one current (c{c} is planes {}..{})", plane + count, cur.base, cur.base + cur.count);
        }
        let k = plane - cur.base;
        self.currents[c].read = true;
        Ok((c, k))
    }

    fn write(&mut self, live: &mut BTreeMap<u32, usize>, plane: u32, count: u32, init: bool, row: usize) -> Result<(usize, u32)> {
        if init {
            return Ok((self.define(live, plane, count, Init::Written).0, 0));
        }
        let c = self.read(live, plane, count, row)?;
        Ok(c)
    }

    /// "c<i>" or "c<i>[k]"
    fn name(&self, operand: Option<(usize, u32)>) -> J {
        match operand {
            None => J::Null,
            Some((c, 0)) => s(format!("c{c}")),
            Some((c, k)) => s(format!("c{c}[{k}]")),
        }
    }
}

/// The rows in the order main.json lists them: the sources first (in program order), each clear
/// that initializes a current just before the first row that uses that current (the clears of
/// unused currents at the end), and every binding that repeats the binding in force dropped;
/// unused plane operands of the symjit rows set to 0. Equivalent to `rows`: the runner runs the
/// sources before every other row, and a current's planes hold no other live current between its
/// clear and its first use.
fn canonical(rows: &[Row], an: &Analysis) -> Vec<Row> {
    let mut out: Vec<Row> = rows.iter().zip(&an.initializer).filter(|(r, i)| **i && matches!(r, Row::Source { .. })).map(|(r, _)| r.clone()).collect();
    let mut cleared = vec![false; an.currents.len()];
    let mut clear = |c: usize, out: &mut Vec<Row>| {
        if an.currents[c].init == Init::Zero && !cleared[c] {
            cleared[c] = true;
            out.push(Row::Clear { base: an.currents[c].base, count: an.currents[c].count });
        }
    };
    let mut binding: Option<&Row> = None;
    for (n, row) in rows.iter().enumerate() {
        if an.initializer[n] {
            continue;
        }
        if let Row::Binding { .. } = row {
            if binding != Some(row) {
                binding = Some(row);
                out.push(row.clone());
            }
            continue;
        }
        for (c, _) in an.operands[n].iter().flatten() {
            clear(*c, &mut out);
        }
        let mut row = row.clone();
        if let Row::Symjit { p0, p1, .. } = &mut row {
            if an.operands[n][0].is_none() {
                *p0 = 0;
            }
            if an.operands[n][1].is_none() {
                *p1 = 0;
            }
        }
        out.push(row);
    }
    for c in 0..an.currents.len() {
        clear(c, &mut out);
    }
    out
}

fn input_code(c: u32) -> Result<String> {
    let (kind, which, im, n) = (c & 7, (c >> 3) & 1, (c >> 4) & 1, c >> 5);
    let part = if im == 1 { "im" } else { "re" };
    Ok(match kind {
        B_PARENT => format!("{part} in[{which}][{n}]"),
        B_DESTINATION if which == 0 => format!("{part} out[{n}]"),
        B_MOMENTUM if im == 0 => format!("momenta[{which}][{n}]"),
        B_SCALE if im == 0 && which == 0 => format!("s{n}"),
        B_FACTOR if which == 0 && n == 0 => format!("{part} factor"),
        _ => bail!("binding input code {c:#x}: not one pac2json writes"),
    })
}

fn parse_input_code(text: &str) -> Result<u32> {
    let bad = || anyhow!("binding input `{text}`");
    let (part, rest) = match text.split_once(' ') {
        Some((p @ ("re" | "im"), r)) => (Some(p), r),
        _ => (None, text),
    };
    let im = u32::from(part == Some("im"));
    let index = |r: &str, prefix: &str| -> Result<u32> {
        r.strip_prefix(prefix).and_then(|t| t.strip_suffix(']')).ok_or_else(bad)?.parse().map_err(|_| bad())
    };
    Ok(match (part, rest) {
        (Some(_), r) if r.starts_with("in[0][") => B_PARENT | im << 4 | index(r, "in[0][")? << 5,
        (Some(_), r) if r.starts_with("in[1][") => B_PARENT | 1 << 3 | im << 4 | index(r, "in[1][")? << 5,
        (Some(_), r) if r.starts_with("out[") => B_DESTINATION | im << 4 | index(r, "out[")? << 5,
        (Some(_), "factor") => B_FACTOR | im << 4,
        (None, r) if r.starts_with("momenta[0][") => B_MOMENTUM | index(r, "momenta[0][")? << 5,
        (None, r) if r.starts_with("momenta[1][") => B_MOMENTUM | 1 << 3 | index(r, "momenta[1][")? << 5,
        (None, r) if r.starts_with('s') => B_SCALE | r[1..].parse::<u32>().map_err(|_| bad())? << 5,
        _ => bail!(bad()),
    })
}

/// p<l> terms of a form as text: "p0 + p2 - 2.0 p3"
fn form_text(slots: &[u32], coefficients: &[f64]) -> String {
    let mut text = String::new();
    for (k, (&leg, &c)) in slots.iter().zip(coefficients).enumerate() {
        let (negative, magnitude) = (c < 0.0, c.abs());
        if k == 0 {
            if negative {
                text.push('-');
            }
        } else {
            text.push_str(if negative { " - " } else { " + " });
        }
        if magnitude != 1.0 {
            let _ = write!(text, "{} ", number(magnitude));
        }
        let _ = write!(text, "p{leg}");
    }
    if text.is_empty() { "0".into() } else { text }
}

struct Info {
    /// per process key: the runtime's particles, helicity configurations, colour components (JSON)
    particles: BTreeMap<String, Value>,
    helicities: BTreeMap<String, Value>,
    colours: BTreeMap<String, Value>,
    expressions: BTreeMap<String, String>,
}

fn value_to_j(v: &Value) -> J {
    match v {
        Value::Null => J::Null,
        Value::Bool(b) => J::B(*b),
        Value::Number(n) => n.as_i64().map(J::I).unwrap_or_else(|| J::F(n.as_f64().unwrap_or(f64::NAN))),
        Value::String(t) => s(t.clone()),
        Value::Array(a) => J::A(a.iter().map(value_to_j).collect()),
        Value::Object(m) => J::O(m.iter().map(|(k, x)| (k.clone(), value_to_j(x))).collect()),
    }
}

fn program_json(p: &Process, c: &Colour, helicity_ids: &[String], kernels: &[String], info: &Info, parameters_here: bool) -> Result<J> {
    let t = &c.template;
    let red = &c.reduction;
    let rows = program::decode(&t.words, &t.width)?;
    let an = Analysis::of(&rows)?;
    let q = |f: u32| format!("q{f}");
    let x = |f: u32| format!("x{f}");
    // the amplitudes: the closures add into them, the reduction reads them
    let mut amplitude_planes: BTreeSet<u32> = red.planes.iter().copied().collect();
    amplitude_planes.extend(rows.iter().filter_map(|r| if let Row::Closure { amplitude, .. } = r { Some(*amplitude) } else { None }));
    let amplitude_planes: Vec<u32> = amplitude_planes.into_iter().collect();
    let a = |plane: u32| format!("a{}", amplitude_planes.binary_search(&plane).unwrap());
    // the bindings, distinct in order of first use
    let mut bindings: Vec<&Row> = Vec::new();
    // the roles of the scales
    let mut roles: BTreeMap<u32, BTreeSet<&str>> = BTreeMap::new();
    let mut initial: BTreeMap<usize, J> = BTreeMap::new();
    let mut body = Vec::new();
    let mut binding = 0usize;
    for (n, row) in rows.iter().enumerate() {
        let ops = &an.operands[n];
        let name = |k: usize| an.name(ops.get(k).copied().flatten());
        let mode = |init: bool| s(if init { "=" } else { "+=" });
        if an.initializer[n] {
            match *row {
                Row::Source { form, factor, mass, code, .. } => {
                    roles.entry(mass).or_default().insert("mass");
                    initial.insert(ops[0].unwrap().0, source_json(form, factor, mass, code, t));
                }
                _ => {
                    initial.insert(ops[0].unwrap().0, u(0));
                }
            }
            continue;
        }
        body.push(match row {
            Row::Clear { count, .. } => o(vec![("op", s("clear")), ("current", name(0)), ("components", u(*count))]),
            Row::Contribution { op, init, slot, f0, f1, factor, .. } => {
                roles.entry(*slot).or_default().insert("coupling");
                o(vec![
                    ("op", s("vertex")),
                    ("type", s(contribution_type(*op))),
                    ("out", name(2)),
                    ("mode", mode(*init)),
                    ("in", J::A(vec![name(0), name(1)])),
                    ("momenta", strs(&[q(*f0), q(*f1)])),
                    ("coupling", s(format!("s{slot}"))),
                    ("factor", s(x(*factor))),
                ])
            }
            Row::Propagator { op, form, factor, .. } => o(vec![
                ("op", s("propagator")),
                ("type", s(propagator_type(*op))),
                ("current", name(0)),
                ("momentum", s(q(*form))),
                ("factor", s(x(*factor))),
            ]),
            Row::Identity { count, factor, .. } => o(vec![
                ("op", s("scale")),
                ("current", name(0)),
                ("components", u(*count)),
                ("factor", s(x(*factor))),
            ]),
            Row::Closure { amplitude, factor, first, count, .. } => o(vec![
                ("op", s("closure")),
                ("amplitude", s(a(*amplitude))),
                ("in", J::A(vec![name(0), name(1)])),
                ("components", u(*count)),
                ("component_factors", strs(&(0..*count).map(|k| x(first + k)).collect::<Vec<_>>())),
                ("factor", s(x(*factor))),
            ]),
            Row::Source { .. } => unreachable!("every source is an initializer"),
            Row::Binding { inputs, .. } => {
                for c in inputs {
                    if c & 7 == B_SCALE {
                        roles.entry(c >> 5).or_default().insert("kernel input");
                    }
                }
                binding = match bindings.iter().position(|b| *b == row) {
                    Some(k) => k,
                    None => {
                        bindings.push(row);
                        bindings.len() - 1
                    }
                };
                continue;
            }
            Row::Symjit { init, f0, f1, factor, .. } => o(vec![
                ("op", s("kernel")),
                ("binding", s(format!("b{binding}"))),
                ("out", name(2)),
                ("mode", mode(*init)),
                ("in", J::A(vec![name(0), name(1)])),
                ("momenta", strs(&[q(*f0), q(*f1)])),
                ("factor", s(x(*factor))),
            ]),
        });
    }
    // inputs
    let crossing = match &p.crossing {
        None => J::Null,
        Some(c) => J::A(c.iter().map(|(leg, sign)| o(vec![("leg", u(*leg)), ("sign", J::F(*sign))])).collect()),
    };
    let forms = J::A(
        (0..t.form_terms.len() / 2)
            .map(|f| {
                let (first, n) = (t.form_terms[2 * f] as usize, t.form_terms[2 * f + 1] as usize);
                let (slots, coefficients) = (&t.term_slots[first..first + n], &t.term_coefficients[first..first + n]);
                o(vec![
                    ("name", s(q(f as u32))),
                    ("value", s(form_text(slots, coefficients))),
                    ("terms", J::A(slots.iter().zip(coefficients).map(|(l, c)| J::A(vec![u(*l), J::F(*c)])).collect())),
                ])
            })
            .collect(),
    );
    let scales = J::A(
        (0..t.scale_re.len())
            .map(|k| {
                let role: Vec<String> = roles.get(&(k as u32)).map(|r| r.iter().map(|x| x.to_string()).collect()).unwrap_or_default();
                o(vec![("name", s(format!("s{k}"))), ("value", cx(t.scale_re[k], t.scale_im[k])), ("used_as", strs(&role))])
            })
            .collect(),
    );
    let factors = J::A(
        (0..t.factor_re.len()).map(|k| o(vec![("name", s(x(k as u32))), ("value", cx(t.factor_re[k], t.factor_im[k]))])).collect(),
    );
    let mut inputs = vec![
        ("legs", u(p.legs)),
        ("particles", particles_json(info.particles.get(&p.key))),
        ("crossing", crossing),
        ("forms", forms),
        ("scales", scales),
        ("factors", factors),
        (
            "alpha_s",
            if p.alpha_s_name.is_empty() {
                J::Null
            } else {
                o(vec![("name", s(p.alpha_s_name.clone())), ("value", J::F(p.alpha_s)), ("power", u(p.qcd_power))])
            },
        ),
    ];
    if parameters_here {
        inputs.push(("parameters", parameters_json(&p.parameters)));
    }
    // output
    let configurations: BTreeMap<String, Value> = info
        .helicities
        .get(&p.key)
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|h| Some((h.get("id")?.as_str()?.to_string(), h.clone()))).collect())
        .unwrap_or_default();
    let helicities = J::A(
        helicity_ids
            .iter()
            .enumerate()
            .map(|(k, id)| {
                let legs = configurations.get(id).and_then(|h| h.get("helicities")).map(value_to_j).unwrap_or(J::Null);
                o(vec![("output", u(k as i64)), ("id", s(id.clone())), ("legs", legs)])
            })
            .collect(),
    );
    let output = o(vec![
        ("count", u(p.helicities)),
        ("group_factor", J::A(vec![u(p.factor.0), u(p.factor.1)])),
        ("helicities", helicities),
    ]);
    // currents, amplitudes, bindings, reduction
    let currents_json = J::A(
        an.currents
            .iter()
            .enumerate()
            .map(|(k, cur)| {
                let mut v = vec![
                    ("name", s(format!("c{k}"))),
                    ("components", u(cur.count)),
                    ("planes", J::A(vec![u(cur.base), u(cur.count)])),
                    ("momentum", cur.momentum.map_or(J::Null, |f| s(q(f)))),
                    (
                        "initial",
                        s(match cur.init {
                            Init::Zero => "zero",
                            Init::Source(_) => "source",
                            Init::Written => "written",
                            Init::Arena => "arena",
                        }),
                    ),
                ];
                if let Init::Source(_) = cur.init {
                    v.push(("source", initial.remove(&k).unwrap_or(J::Null)));
                }
                o(v)
            })
            .collect(),
    );
    let amplitudes = J::A(
        amplitude_planes
            .iter()
            .enumerate()
            .map(|(k, plane)| o(vec![("name", s(format!("a{k}"))), ("plane", u(*plane)), ("initial", u(0))]))
            .collect(),
    );
    let bindings_json = J::A(
        bindings
            .iter()
            .enumerate()
            .map(|(k, b)| {
                let Row::Binding { kernel, direct, inputs, outputs } = b else { unreachable!() };
                let names: Vec<String> = inputs.iter().map(|c| input_code(*c)).collect::<Result<_>>()?;
                Ok(o(vec![
                    ("name", s(format!("b{k}"))),
                    ("kernel", s(format!("k{kernel}"))),
                    ("sha256", s(kernels.get(*kernel as usize).cloned().unwrap_or_default())),
                    ("direct", J::B(*direct)),
                    ("inputs", strs(&names)),
                    ("outputs", J::A(outputs.iter().map(|c| u(*c)).collect())),
                ]))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    let reduction = J::A(
        (0..red.offsets.len() - 1)
            .map(|h| {
                let terms = (red.offsets[h]..red.offsets[h + 1])
                    .map(|k| J::A(vec![s(a(red.planes[k as usize])), J::F(red.weights[k as usize])]))
                    .collect();
                o(vec![("output", u(h as i64)), ("terms", J::A(terms))])
            })
            .collect(),
    );
    let colour_info = info
        .colours
        .get(&p.key)
        .and_then(|v| v.as_array())
        .and_then(|a| a.iter().find(|x| x.get("id").and_then(|i| i.as_str()) == Some(c.id.as_str())))
        .map(|x| o(vec![("kind", value_to_j(&x["kind"])), ("word", value_to_j(&x["word"])), ("coefficient", value_to_j(&x["coefficient"]))]))
        .unwrap_or(J::Null);
    let layout = o(vec![
        ("points_per_run", u(t.tile)),
        ("current_planes", u(t.current_planes)),
        ("amplitude_planes", u(t.amplitude_planes)),
        ("momentum_planes", u(t.momentum_planes)),
        ("zero_amplitude", J::A(t.zero_amplitude.iter().map(|v| u(*v)).collect())),
        ("download_amplitude", J::A(t.download_amplitude.iter().map(|v| u(*v)).collect())),
        ("field_bits", J::A(t.width.iter().map(|v| u(*v)).collect())),
    ]);
    Ok(o(vec![
        ("process", s(p.key.clone())),
        ("expression", info.expressions.get(&p.key).map_or(J::Null, |e| s(e.clone()))),
        ("colour", s(c.id.clone())),
        ("colour_component", colour_info),
        ("inputs", o(inputs)),
        ("output", output),
        ("currents", currents_json),
        ("amplitudes", amplitudes),
        ("bindings", bindings_json),
        ("body", J::A(body)),
        ("reduction", reduction),
        ("layout", layout),
    ]))
}

/// per external leg: its index, particle, PDG code and side
fn particles_json(particles: Option<&Value>) -> J {
    let Some(list) = particles.and_then(|v| v.as_array()) else { return J::Null };
    J::A(
        list.iter()
            .map(|x| {
                o(vec![
                    ("leg", value_to_j(&x["index"])),
                    ("particle", value_to_j(&x["particle"])),
                    ("pdg", value_to_j(&x["pdg"])),
                    ("side", value_to_j(&x["side"])),
                ])
            })
            .collect(),
    )
}

fn source_json(form: u32, factor: u32, mass: u32, code: u32, t: &Template) -> J {
    let family = (code & 7) as usize;
    let orientation = (code >> 3) & 3;
    let helicity = ((code >> 5) & 7) as i64 - 2;
    let chirality = ((code >> 8) & 3) as i64 - 1;
    let mass_value = t.scale_re.get(mass as usize).copied().unwrap_or(f64::NAN);
    let side = if orientation == 0 { "particle" } else { "antiparticle" };
    let name = match family {
        0 => "scalar".to_string(),
        3 if mass_value == 0.0 => "vector (massless)".to_string(),
        3 => "vector (massive)".to_string(),
        _ => format!("{} {side}", FAMILIES[family]),
    };
    let leg = match (t.form_terms.get(2 * form as usize), t.form_terms.get(2 * form as usize + 1)) {
        (Some(&first), Some(1)) => Some(t.term_slots[first as usize]),
        _ => None,
    };
    o(vec![
        ("wavefunction", s(name)),
        ("family", s(FAMILIES[family])),
        ("orientation", s(side)),
        ("leg", leg.map_or(J::Null, u)),
        ("momentum", s(format!("q{form}"))),
        ("helicity", u(helicity)),
        ("chirality", u(chirality)),
        ("mass", s(format!("s{mass}"))),
        ("factor", s(format!("x{factor}"))),
    ])
}

fn parameters_json(parameters: &[(String, f64, f64)]) -> J {
    J::A(parameters.iter().map(|(n, re, im)| o(vec![("name", s(n.clone())), ("value", cx(*re, *im))])).collect())
}

fn examples_json(file: &File) -> J {
    let mut programs = Vec::new();
    for p in &file.processes {
        let (n, h) = (p.legs as usize, p.helicities as usize);
        for c in &p.colours {
            let points: Vec<J> = (0..c.reference_points as usize)
                .map(|i| {
                    let momenta = (0..n)
                        .map(|leg| J::A((0..4).map(|k| J::F(c.reference_momenta[(i * n + leg) * 4 + k])).collect()))
                        .collect();
                    let values = &c.reference_values[i * h..(i + 1) * h];
                    o(vec![
                        ("momenta", J::A(momenta)),
                        ("outputs", J::A(values.iter().map(|v| J::F(*v)).collect())),
                        ("total", J::F(values.iter().sum())),
                    ])
                })
                .collect();
            programs.push(o(vec![("process", s(p.key.clone())), ("colour", s(c.id.clone())), ("points", J::A(points))]));
        }
    }
    o(vec![
        (
            "description",
            s("per program, phase-space points and the CPU's outputs (RustiCol, f64): momenta[leg] = [E, px, py, pz], the caller's legs before \
               the crossing; outputs[h] as main.json's output.helicities, group factor applied; total = Σ outputs (in order). Momenta: two \
               incoming beams along z at sqrt(s) = 1000, the rest by RAMBO from a fixed seed"),
        ),
        ("programs", J::A(programs)),
    ])
}

// ---- reading a pac2json folder back --------------------------------------------------------------

fn get<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key).ok_or_else(|| anyhow!("no `{key}`"))
}
fn get_u32(v: &Value, key: &str) -> Result<u32> {
    as_u32(get(v, key)?).with_context(|| format!("`{key}`"))
}
fn get_str<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    get(v, key)?.as_str().ok_or_else(|| anyhow!("`{key}` is not a string"))
}
fn get_arr<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    get(v, key)?.as_array().ok_or_else(|| anyhow!("`{key}` is not an array"))
}
fn as_f64(v: &Value) -> Result<f64> {
    v.as_f64().ok_or_else(|| anyhow!("{v} is not a number"))
}
fn as_u32(v: &Value) -> Result<u32> {
    v.as_u64().and_then(|x| u32::try_from(x).ok()).ok_or_else(|| anyhow!("{v} is not a count"))
}
fn as_i64(v: &Value) -> Result<i64> {
    v.as_i64().ok_or_else(|| anyhow!("{v} is not an integer"))
}
fn complex(v: &Value) -> Result<(f64, f64)> {
    match v.as_array().map(|a| a.as_slice()) {
        Some([re, im]) => Ok((as_f64(re)?, as_f64(im)?)),
        _ => bail!("{v} is not [re, im]"),
    }
}
/// the index of a name `<prefix><i>` (and the component of `<prefix><i>[k]`)
fn named(text: &str, prefix: &str) -> Result<(u32, u32)> {
    let rest = text.strip_prefix(prefix).ok_or_else(|| anyhow!("`{text}` is not a {prefix} name"))?;
    let (index, component) = match rest.split_once('[') {
        Some((i, c)) => (i, c.strip_suffix(']').ok_or_else(|| anyhow!("`{text}`"))?),
        None => (rest, "0"),
    };
    Ok((index.parse().map_err(|_| anyhow!("`{text}`"))?, component.parse().map_err(|_| anyhow!("`{text}`"))?))
}
fn index_of(v: &Value, prefix: &str) -> Result<u32> {
    Ok(named(v.as_str().ok_or_else(|| anyhow!("{v} is not a {prefix} name"))?, prefix)?.0)
}
fn two<'a>(v: &'a Value, key: &str) -> Result<(&'a Value, &'a Value)> {
    match get_arr(v, key)?.as_slice() {
        [a, b] => Ok((a, b)),
        _ => bail!("`{key}` is not a pair"),
    }
}

struct Reader {
    /// first plane per current, amplitude plane per amplitude
    current_base: Vec<u32>,
    amplitude_plane: Vec<u32>,
}

impl Reader {
    fn current(&self, v: &Value) -> Result<u32> {
        if v.is_null() {
            return Ok(0);
        }
        let text = v.as_str().ok_or_else(|| anyhow!("{v} is not a current"))?;
        let (k, c) = named(text, "c")?;
        Ok(self.current_base.get(k as usize).ok_or_else(|| anyhow!("no current {text}"))? + c)
    }
    fn amplitude(&self, v: &Value) -> Result<u32> {
        let text = v.as_str().ok_or_else(|| anyhow!("{v} is not an amplitude"))?;
        Ok(*self.amplitude_plane.get(named(text, "a")?.0 as usize).ok_or_else(|| anyhow!("no amplitude {text}"))?)
    }
}

fn source_code(v: &Value) -> Result<u32> {
    let family = get_str(v, "family")?;
    let family = FAMILIES.iter().position(|f| *f == family).ok_or_else(|| anyhow!("source family {family}"))? as u32;
    let orientation = u32::from(get_str(v, "orientation")? == "antiparticle");
    let helicity = as_i64(get(v, "helicity")?)? + 2;
    let chirality = as_i64(get(v, "chirality")?)? + 1;
    Ok(family | orientation << 3 | (helicity as u32) << 5 | (chirality as u32) << 8)
}

fn source_row(dst: u32, v: &Value) -> Result<Row> {
    Ok(Row::Source {
        dst,
        form: index_of(get(v, "momentum")?, "q")?,
        factor: index_of(get(v, "factor")?, "x")?,
        mass: index_of(get(v, "mass")?, "s")?,
        code: source_code(v)?,
    })
}

fn read_program(v: &Value, parameters: &[(String, f64, f64)]) -> Result<(Process, Colour)> {
    let inputs = get(v, "inputs")?;
    let output = get(v, "output")?;
    let layout = get(v, "layout")?;
    let currents = get_arr(v, "currents")?;
    let reader = Reader {
        current_base: currents.iter().map(|c| as_u32(&get_arr(c, "planes")?[0])).collect::<Result<_>>()?,
        amplitude_plane: get_arr(v, "amplitudes")?.iter().map(|a| get_u32(a, "plane")).collect::<Result<_>>()?,
    };
    // the rows: the sources, then the body with each current's clear before its first use
    let mut rows = Vec::new();
    let mut zero: Vec<Option<Row>> = Vec::new();
    for (k, c) in currents.iter().enumerate() {
        let base = reader.current_base[k];
        zero.push(None);
        match get_str(c, "initial")? {
            "source" => rows.push(source_row(base, get(c, "source")?).with_context(|| format!("current c{k}"))?),
            "zero" => zero[k] = Some(Row::Clear { base, count: get_u32(c, "components")? }),
            "written" | "arena" => {}
            other => bail!("current c{k}: initial {other}"),
        }
    }
    let bindings = get_arr(v, "bindings")?;
    let mut binding: Option<usize> = None;
    for (n, i) in get_arr(v, "body")?.iter().enumerate() {
        let factor = || index_of(get(i, "factor")?, "x");
        let init = || -> Result<bool> {
            match get_str(i, "mode")? {
                "=" => Ok(true),
                "+=" => Ok(false),
                m => bail!("mode {m}"),
            }
        };
        let row = match get_str(i, "op")? {
            "clear" => Row::Clear { base: reader.current(get(i, "current")?)?, count: get_u32(i, "components")? },
            "vertex" => {
                let ty = get_str(i, "type")?;
                let op = (2..=6).find(|&op| contribution_type(op) == ty).ok_or_else(|| anyhow!("vertex type {ty}"))?;
                let (p0, p1) = two(i, "in")?;
                let (f0, f1) = two(i, "momenta")?;
                Row::Contribution {
                    op,
                    init: init()?,
                    slot: index_of(get(i, "coupling")?, "s")?,
                    p0: reader.current(p0)?,
                    p1: reader.current(p1)?,
                    dst: reader.current(get(i, "out")?)?,
                    f0: index_of(f0, "q")?,
                    f1: index_of(f1, "q")?,
                    factor: factor()?,
                }
            }
            "propagator" => {
                let ty = get_str(i, "type")?;
                let op = (7..=9).find(|&op| propagator_type(op) == ty).ok_or_else(|| anyhow!("propagator type {ty}"))?;
                Row::Propagator { op, base: reader.current(get(i, "current")?)?, form: index_of(get(i, "momentum")?, "q")?, factor: factor()? }
            }
            "scale" => Row::Identity { base: reader.current(get(i, "current")?)?, count: get_u32(i, "components")?, factor: factor()? },
            "closure" => {
                let (p0, p1) = two(i, "in")?;
                let count = get_u32(i, "components")?;
                let cf = get_arr(i, "component_factors")?;
                let first = cf.first().map(|f| index_of(f, "x")).transpose()?.unwrap_or(0);
                if cf.len() != count as usize || cf.iter().enumerate().any(|(k, f)| index_of(f, "x").ok() != Some(first + k as u32)) {
                    bail!("body[{n}]: component_factors: {count} consecutive factors expected");
                }
                Row::Closure { p0: reader.current(p0)?, p1: reader.current(p1)?, amplitude: reader.amplitude(get(i, "amplitude")?)?, factor: factor()?, first, count }
            }
            "source" => source_row(reader.current(get(i, "current")?)?, i)?,
            "kernel" => {
                let b = index_of(get(i, "binding")?, "b")? as usize;
                if binding != Some(b) {
                    let bv = bindings.get(b).ok_or_else(|| anyhow!("body[{n}]: no binding b{b}"))?;
                    rows.push(Row::Binding {
                        kernel: index_of(get(bv, "kernel")?, "k")?,
                        direct: get(bv, "direct")?.as_bool().ok_or_else(|| anyhow!("b{b}: direct"))?,
                        inputs: get_arr(bv, "inputs")?.iter().map(|c| parse_input_code(c.as_str().unwrap_or(""))).collect::<Result<_>>()?,
                        outputs: get_arr(bv, "outputs")?.iter().map(as_u32).collect::<Result<_>>()?,
                    });
                    binding = Some(b);
                }
                let (p0, p1) = two(i, "in")?;
                let (f0, f1) = two(i, "momenta")?;
                Row::Symjit {
                    init: init()?,
                    p0: reader.current(p0)?,
                    p1: reader.current(p1)?,
                    dst: reader.current(get(i, "out")?)?,
                    f0: index_of(f0, "q")?,
                    f1: index_of(f1, "q")?,
                    factor: factor()?,
                }
            }
            op => bail!("body[{n}]: op {op}"),
        };
        let operands: Vec<&Value> = match get_str(i, "op")? {
            "vertex" | "kernel" => {
                let (p0, p1) = two(i, "in")?;
                vec![p0, p1, get(i, "out")?]
            }
            "closure" => {
                let (p0, p1) = two(i, "in")?;
                vec![p0, p1]
            }
            _ => vec![get(i, "current")?],
        };
        for operand in operands.into_iter().filter_map(|x| x.as_str()) {
            let k = named(operand, "c")?.0 as usize;
            if let Some(clear) = zero.get_mut(k).and_then(Option::take) {
                rows.push(clear);
            }
        }
        rows.push(row);
    }
    rows.extend(zero.into_iter().flatten());
    let width: [u8; 8] = get_arr(layout, "field_bits")?
        .iter()
        .map(|b| as_u32(b).map(|b| b as u8))
        .collect::<Result<Vec<u8>>>()?
        .try_into()
        .map_err(|_| anyhow!("field_bits: 8 values"))?;
    let words = program::encode(&rows, &width)?;
    // the tables
    let (mut scale_re, mut scale_im, mut factor_re, mut factor_im) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for sc in get_arr(inputs, "scales")? {
        let (re, im) = complex(get(sc, "value")?)?;
        scale_re.push(re);
        scale_im.push(im);
    }
    for f in get_arr(inputs, "factors")? {
        let (re, im) = complex(get(f, "value")?)?;
        factor_re.push(re);
        factor_im.push(im);
    }
    let (mut form_terms, mut term_slots, mut term_coefficients) = (Vec::new(), Vec::new(), Vec::new());
    for f in get_arr(inputs, "forms")? {
        let terms = get_arr(f, "terms")?;
        form_terms.extend([term_slots.len() as u32, terms.len() as u32]);
        for t in terms {
            match t.as_array().map(|a| a.as_slice()) {
                Some([leg, c]) => {
                    term_slots.push(as_u32(leg)?);
                    term_coefficients.push(as_f64(c)?);
                }
                _ => bail!("form term {t}"),
            }
        }
    }
    let runs = |key: &str| -> Result<Vec<u32>> { get_arr(layout, key)?.iter().map(as_u32).collect() };
    let legs = get_u32(inputs, "legs")?;
    let template = Template {
        tile: get_u32(layout, "points_per_run")?,
        words,
        width,
        current_planes: get_u32(layout, "current_planes")?,
        amplitude_planes: get_u32(layout, "amplitude_planes")?,
        momentum_planes: get_u32(layout, "momentum_planes")?,
        scale_re,
        scale_im,
        factor_re,
        factor_im,
        zero_amplitude: runs("zero_amplitude")?,
        download_amplitude: runs("download_amplitude")?,
        external_count: legs,
        form_terms,
        term_slots,
        term_coefficients,
    };
    let (mut offsets, mut planes, mut weights) = (vec![0u32], Vec::new(), Vec::new());
    for (h, r) in get_arr(v, "reduction")?.iter().enumerate() {
        if get_u32(r, "output")? as usize != h {
            bail!("reduction[{h}] is not output {h}");
        }
        for t in get_arr(r, "terms")? {
            match t.as_array().map(|a| a.as_slice()) {
                Some([a, w]) => {
                    planes.push(reader.amplitude(a)?);
                    weights.push(as_f64(w)?);
                }
                _ => bail!("reduction term {t}"),
            }
        }
        offsets.push(planes.len() as u32);
    }
    let alpha = get(inputs, "alpha_s")?;
    let (alpha_s_name, alpha_s, qcd_power) = if alpha.is_null() {
        (String::new(), 0.0, 0)
    } else {
        (get_str(alpha, "name")?.to_string(), as_f64(get(alpha, "value")?)?, as_i64(get(alpha, "power")?)? as i32)
    };
    let parameters = match inputs.get("parameters") {
        Some(p) => read_parameters(p)?,
        None => parameters.to_vec(),
    };
    let crossing = match get(inputs, "crossing")? {
        Value::Null => None,
        c => Some(
            c.as_array()
                .ok_or_else(|| anyhow!("crossing"))?
                .iter()
                .map(|x| Ok((get_u32(x, "leg")?, as_f64(get(x, "sign")?)?)))
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    let factor = match get_arr(output, "group_factor")?.as_slice() {
        [n, d] => (as_i64(n)?, as_i64(d)?),
        _ => bail!("group_factor"),
    };
    let process = Process {
        key: get_str(v, "process")?.to_string(),
        legs,
        helicities: get_u32(output, "count")?,
        alpha_s_name,
        alpha_s,
        qcd_power,
        parameters,
        factor,
        crossing,
        colours: Vec::new(),
    };
    let colour = Colour {
        id: get_str(v, "colour")?.to_string(),
        template,
        reduction: Reduction { offsets, planes, weights },
        reference_points: 0,
        reference_momenta: Vec::new(),
        reference_values: Vec::new(),
    };
    Ok((process, colour))
}

fn read_parameters(v: &Value) -> Result<Vec<(String, f64, f64)>> {
    v.as_array()
        .ok_or_else(|| anyhow!("parameters"))?
        .iter()
        .map(|p| {
            let (re, im) = complex(get(p, "value")?)?;
            Ok((get_str(p, "name")?.to_string(), re, im))
        })
        .collect()
}

/// The `.pacuda` contents a pac2json folder describes (the program rows in main.json's order).
fn read_folder(dir: &Path, kernel: u32, batch: u32) -> Result<File> {
    let read = |name: &str| -> Result<Value> {
        let path = dir.join(name);
        let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
        serde_json::from_str(&text).with_context(|| path.display().to_string())
    };
    let main = read("main.json")?;
    let examples = read("examples.json")?;
    let configs = get(&main, "configs")?;
    if get_u32(configs, "format")? != FORMAT {
        bail!("main.json: format {}, this pac2json reads {FORMAT}", get(configs, "format")?);
    }
    let parameters = match configs.get("parameters") {
        Some(p) => read_parameters(p)?,
        None => Vec::new(),
    };
    let symjit_sha256 = get_arr(&main, "kernels")?.iter().map(|k| Ok(get_str(k, "sha256")?.to_string())).collect::<Result<Vec<_>>>()?;
    let example_programs = get_arr(&examples, "programs")?;
    let mut processes: Vec<Process> = Vec::new();
    for (n, p) in get_arr(&main, "programs")?.iter().enumerate() {
        let (process, mut colour) = read_program(p, &parameters).with_context(|| format!("main.json: programs[{n}]"))?;
        let e = example_programs.get(n).ok_or_else(|| anyhow!("examples.json: no programs[{n}]"))?;
        if get_str(e, "process")? != process.key || get_str(e, "colour")? != colour.id {
            bail!("examples.json: programs[{n}] is not {} {}", process.key, colour.id);
        }
        for point in get_arr(e, "points")? {
            for m in get_arr(point, "momenta")? {
                for x in m.as_array().ok_or_else(|| anyhow!("momenta"))? {
                    colour.reference_momenta.push(as_f64(x)?);
                }
            }
            for x in get_arr(point, "outputs")? {
                colour.reference_values.push(as_f64(x)?);
            }
            colour.reference_points += 1;
        }
        match processes.last_mut() {
            Some(last) if last.key == process.key => {
                let mut same = process.clone();
                same.colours = last.colours.clone();
                if &same != last {
                    bail!("main.json: programs[{n}]: process {} differs from its previous program's", process.key);
                }
                last.colours.push(colour);
            }
            _ => {
                let mut process = process;
                process.colours.push(colour);
                processes.push(process);
            }
        }
    }
    Ok(File { kernel, batch, symjit_sha256, processes })
}

/// `file` with every program's words in main.json's row order (`canonical`).
fn canonical_file(file: &File) -> Result<File> {
    let mut out = file.clone();
    for p in &mut out.processes {
        for c in &mut p.colours {
            let t = &mut c.template;
            let rows = program::decode(&t.words, &t.width)?;
            let an = Analysis::of(&rows)?;
            let rows = canonical(&rows, &an);
            t.words = program::encode(&rows, &t.width)?;
        }
    }
    Ok(out)
}

// ---- the kernels -------------------------------------------------------------------------------

fn kernels_json(lc: &Path, file: &File, dir: &Path) -> Result<J> {
    record::extract_kernels(lc, &file.symjit_sha256, dir)?;
    let pack = rusticol_core::pacbin::PacbinReader::open(lc.join("evaluators.pacbin")).map_err(|e| anyhow!("{}: {e}", lc.display()))?;
    let mut members: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for member in pack.members() {
        let hex: String = member.sha256().iter().map(|b| format!("{b:02x}")).collect();
        if member.logical_path().ends_with(".plane.symjit") && file.symjit_sha256.contains(&hex) {
            members.entry(hex).or_default().push(member.logical_path().to_string());
        }
    }
    // the shape and the uses of each kernel, from the bindings
    let mut shape: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    let mut uses: BTreeMap<u32, usize> = BTreeMap::new();
    for p in &file.processes {
        for c in &p.colours {
            let mut current = None;
            for row in program::decode(&c.template.words, &c.template.width)? {
                match row {
                    Row::Binding { kernel, ref inputs, ref outputs, .. } => {
                        shape.entry(kernel).or_insert((inputs.len(), outputs.len()));
                        current = Some(kernel);
                    }
                    Row::Symjit { .. } => *uses.entry(current.unwrap_or(u32::MAX)).or_default() += 1,
                    _ => {}
                }
            }
        }
    }
    let mut out = Vec::new();
    for (k, sha) in file.symjit_sha256.iter().enumerate() {
        let bytes = std::fs::metadata(dir.join(format!("{sha}.plane.symjit")))?.len();
        let (inputs, pairs) = shape.get(&(k as u32)).copied().unwrap_or((0, 0));
        out.push(o(vec![
            ("name", s(format!("k{k}"))),
            ("sha256", s(sha.clone())),
            ("path", s(format!("kernels/{sha}.plane.symjit"))),
            ("bytes", u(bytes as i64)),
            ("artifact_members", strs(members.get(sha).map(|v| v.as_slice()).unwrap_or(&[]))),
            ("inputs", u(inputs as i64)),
            ("output_pairs", u(pairs as i64)),
            ("rows", u(*uses.get(&(k as u32)).unwrap_or(&0) as i64)),
            (
                "format",
                s("a symjit plane application (.plane.symjit: the MIR of one executor row, f64); inputs and outputs as the bindings \
                   list them, each output a (re, im) pair"),
            ),
        ]));
    }
    Ok(J::A(out))
}

// ---- main ----------------------------------------------------------------------------------------

fn info_of(lc: &Path, file: &File, parameters: &[(String, f64)]) -> Result<Info> {
    let mut info = Info { particles: BTreeMap::new(), helicities: BTreeMap::new(), colours: BTreeMap::new(), expressions: BTreeMap::new() };
    for p in &file.processes {
        let (runtime, _) = record::load(lc, &p.key, parameters)?;
        let particles = runtime.external_particles().map_err(|e| anyhow!("{}: {e}", p.key))?;
        let helicities = runtime.helicities().map_err(|e| anyhow!("{}: {e}", p.key))?;
        let colours = runtime.color_components().map_err(|e| anyhow!("{}: {e}", p.key))?;
        info.particles.insert(p.key.clone(), serde_json::to_value(particles)?);
        info.helicities.insert(p.key.clone(), serde_json::to_value(helicities)?);
        info.colours.insert(p.key.clone(), serde_json::to_value(colours)?);
    }
    if let Ok(text) = std::fs::read_to_string(lc.join("artifact.json")) {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            for p in v.get("processes").and_then(|x| x.as_array()).into_iter().flatten() {
                if let (Some(id), Some(e)) = (p.get("id").and_then(|x| x.as_str()), p.get("expression").and_then(|x| x.as_str())) {
                    info.expressions.insert(id.to_string(), e.to_string());
                }
            }
        }
    }
    Ok(info)
}

fn convert(lc: &Path, out: &Path, parameters: &[(String, f64)], points: usize, examples: usize) -> Result<()> {
    let (file, helicity_ids) = record::convert_artifact(lc, points, parameters, 0, 0, examples)?;
    std::fs::create_dir_all(out.join("kernels")).with_context(|| out.display().to_string())?;
    let kernels = kernels_json(lc, &file, &out.join("kernels"))?;
    let info = info_of(lc, &file, parameters)?;
    let shared = file.processes.windows(2).all(|w| w[0].parameters == w[1].parameters);
    let mut programs = Vec::new();
    let mut instructions_n = 0;
    for (p, ids) in file.processes.iter().zip(&helicity_ids) {
        for c in &p.colours {
            let program = program_json(p, c, ids, &file.symjit_sha256, &info, !shared).with_context(|| format!("{} {}", p.key, c.id))?;
            if let J::O(m) = &program {
                if let Some((_, J::A(body))) = m.iter().find(|(k, _)| k == "body") {
                    instructions_n += body.len();
                }
            }
            programs.push(program);
        }
    }
    let artifact: Value = std::fs::read_to_string(lc.join("artifact.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
    let lc_abs = std::fs::canonicalize(lc).unwrap_or_else(|_| lc.to_path_buf());
    let mut configs = vec![
        ("generator", s(format!("pac2json {}", env!("CARGO_PKG_VERSION")))),
        ("format", u(FORMAT)),
        ("artifact", s(lc_abs.display().to_string())),
        ("artifact_id", artifact.get("artifact_id").map_or(J::Null, value_to_j)),
        (
            "producer",
            artifact.get("producer").and_then(|p| p.get("version")).and_then(|v| v.as_str()).map_or(J::Null, |v| s(format!("pyamplicol {v}"))),
        ),
        ("model", artifact.get("model").and_then(|m| m.get("name")).map_or(J::Null, value_to_j)),
        ("colour_accuracy", s("lc (leading colour): one program per (process, colour flow)")),
        ("recorded_points", u(points as i64)),
        ("programs", u(programs.len() as i64)),
    ];
    if shared {
        configs.push(("parameters", parameters_json(file.processes.first().map_or(&[][..], |p| &p.parameters))));
    }
    configs.push(("conventions", conventions()));
    configs.push(("instructions", instructions()));
    let main = o(vec![("configs", o(configs)), ("kernels", kernels), ("programs", J::A(programs))]);
    write_json(&out.join("main.json"), &main)?;
    write_json(&out.join("examples.json"), &examples_json(&file))?;
    // the folder reads back as the recording
    let back = read_folder(out, file.kernel, file.batch).context("reading the folder back")?;
    if back != canonical_file(&file)? {
        bail!("{}: the folder does not rebuild the recording", out.display());
    }
    let colours: usize = file.processes.iter().map(|p| p.colours.len()).sum();
    println!(
        "{}: {} processes, {colours} programs ({instructions_n} body instructions), {} symjit kernels, {examples} examples each; reads back as recorded",
        out.display(),
        file.processes.len(),
        file.symjit_sha256.len()
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = (|| -> Result<()> {
        let mut paths = Vec::<PathBuf>::new();
        let (mut out, mut parameters, mut points, mut examples) = (None::<PathBuf>, Vec::new(), 1000usize, 8usize);
        let (mut kernel, mut batch) = (0u32, 0u32);
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let mut value = || it.next().ok_or_else(|| anyhow!("{a} needs a value"));
            match a.as_str() {
                "-o" => out = Some(PathBuf::from(value()?)),
                "-n" => points = value()?.parse()?,
                "-e" => examples = value()?.parse()?,
                "--parameters" => {
                    let path = value()?;
                    let text = std::fs::read_to_string(path).with_context(|| path.clone())?;
                    parameters = record::parse_parameters(&text).with_context(|| path.clone())?;
                }
                "--kernel" => kernel = record::kernel_number(Some(value()?))?,
                "--batch" => batch = value()?.parse()?,
                _ => paths.push(PathBuf::from(a)),
            }
        }
        let out = out.ok_or_else(|| anyhow!("usage: pac2json LC_DIR -o OUT_DIR ... | pac2json pack OUT_DIR -o FILE.pacuda"))?;
        if paths.first().and_then(|p| p.to_str()) == Some("pack") {
            let dir = paths.get(1).ok_or_else(|| anyhow!("pack needs OUT_DIR"))?;
            let file = read_folder(dir, kernel, batch)?;
            file.write(&out)?;
            println!("{}: {} processes from {}", out.display(), file.processes.len(), dir.display());
            return Ok(());
        }
        let lc = paths.first().ok_or_else(|| anyhow!("pac2json needs LC_DIR"))?;
        if examples > points {
            bail!("-e {examples}: at most the recorded points (-n {points})");
        }
        convert(lc, &out, &parameters, points, examples)
    })();
    if let Err(e) = result {
        eprintln!("pac2json: {e:#}");
        std::process::exit(1);
    }
}
