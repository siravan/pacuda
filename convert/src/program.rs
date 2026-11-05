// SPDX-License-Identifier: 0BSD
//! The rows of a recorded plan run: its 64-bit words decoded as the runner decodes them
//! (`decode` in `runner/recurrence_runner.cu`; the encoding is described there and in
//! `runner/pacman_recurrence.h`).

use anyhow::{bail, Result};

/// the `width` indices (`W_P` .. `W_K` of `pacman_recurrence.h`)
pub const W_P: usize = 0;
pub const W_A: usize = 1;
pub const W_F: usize = 2;
pub const W_X: usize = 3;
pub const W_S: usize = 4;
pub const W_K: usize = 5;

/// binding input kinds (`symjit_row`)
pub const B_PARENT: u32 = 0;
pub const B_MOMENTUM: u32 = 1;
pub const B_DESTINATION: u32 = 2;
pub const B_SCALE: u32 = 3;
pub const B_FACTOR: u32 = 4;

#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    /// op 1: current planes `base .. base + count` set to 0
    Clear { base: u32, count: u32 },
    /// ops 2–6: `dst[k] (+)= value_op(p0, p1, forms)[k] × scale[slot] × factor[factor]`
    Contribution { op: u32, init: bool, slot: u32, p0: u32, p1: u32, dst: u32, f0: u32, f1: u32, factor: u32 },
    /// ops 7–9: the current at `base` multiplied by the propagator of momentum form `form`
    Propagator { op: u32, base: u32, form: u32, factor: u32 },
    /// op 10: planes `base .. base + count` multiplied by factor[factor]
    Identity { base: u32, count: u32, factor: u32 },
    /// op 11: amplitude[amplitude] += Σ_k p0[k] p1[k] factor[first + k] × factor[factor]
    Closure { p0: u32, p1: u32, amplitude: u32, factor: u32, first: u32, count: u32 },
    /// op 12: the external wavefunction of template `code` (family, orientation, helicity,
    /// chirality) from momentum form `form` and mass scale[mass], × factor[factor], into `dst`
    Source { dst: u32, form: u32, factor: u32, mass: u32, code: u32 },
    /// op 13: a symjit-compiled row, by the binding of the last op 14
    Symjit { init: bool, p0: u32, p1: u32, dst: u32, f0: u32, f1: u32, factor: u32 },
    /// op 14: the binding of the symjit rows that follow: kernel, input codes, destination
    /// components of the output pairs
    Binding { kernel: u32, direct: bool, inputs: Vec<u32>, outputs: Vec<u32> },
}

/// (components read from parent 0, from parent 1, written) of a contribution op
pub fn contribution_shape(op: u32) -> (u32, u32, u32) {
    let c0 = match op {
        4 => 6,
        5 | 6 => 4,
        _ => 2,
    };
    let cd = match op {
        4 | 6 => 4,
        5 => 6,
        _ => 2,
    };
    (c0, 4, cd)
}

pub fn decode(words: &[u64], width: &[u8; 8]) -> Result<Vec<Row>> {
    let mut rows = Vec::new();
    let mut n = 0;
    while n < words.len() {
        let mut w = words[n];
        if w & 15 == 14 {
            let kernel = ((w >> 4) & 0xffff) as u32;
            let n_in = ((w >> 20) & 0xff) as usize;
            let pairs = ((w >> 28) & 0xff) as usize;
            let direct = (w >> 36) & 1 == 1;
            let codes: Vec<u32> = (0..n_in + pairs)
                .map(|k| {
                    let d = words.get(n + 1 + k / 2).copied().unwrap_or(0);
                    (if k % 2 == 1 { d >> 32 } else { d & 0xffff_ffff }) as u32
                })
                .collect();
            if n + (n_in + pairs).div_ceil(2) >= words.len() {
                bail!("word {n}: a binding whose data runs past the program");
            }
            rows.push(Row::Binding { kernel, direct, inputs: codes[..n_in].to_vec(), outputs: codes[n_in..].to_vec() });
            n += 1 + (n_in + pairs).div_ceil(2);
            continue;
        }
        let mut take = |bits: u8| -> u32 {
            let v = (w & ((1u64 << bits) - 1)) as u32;
            w >>= bits;
            v
        };
        let op = take(4);
        let (p, a, f, x, s, k) = (width[W_P], width[W_A], width[W_F], width[W_X], width[W_S], width[W_K]);
        let row = match op {
            1 => Row::Clear { base: take(p), count: take(p) },
            2..=6 => Row::Contribution {
                op,
                init: take(1) == 1,
                slot: take(s),
                p0: take(p),
                p1: take(p),
                dst: take(p),
                f0: take(f),
                f1: take(f),
                factor: take(x),
            },
            7..=9 => Row::Propagator { op, base: take(p), form: take(f), factor: take(x) },
            10 => Row::Identity { base: take(p), count: take(k), factor: take(x) },
            11 => Row::Closure { p0: take(p), p1: take(p), amplitude: take(a), factor: take(x), first: take(x), count: take(k) },
            12 => Row::Source { dst: take(p), form: take(f), factor: take(x), mass: take(s), code: take(10) },
            13 => Row::Symjit { init: take(1) == 1, p0: take(p), p1: take(p), dst: take(p), f0: take(f), f1: take(f), factor: take(x) },
            _ => bail!("word {n}: op {op}"),
        };
        rows.push(row);
        n += 1;
    }
    Ok(rows)
}

/// The words of `rows` with field widths `width` (the inverse of `decode`).
pub fn encode(rows: &[Row], width: &[u8; 8]) -> Result<Vec<u64>> {
    let mut words = Vec::new();
    for (n, row) in rows.iter().enumerate() {
        if let Row::Binding { kernel, direct, inputs, outputs } = row {
            if *kernel > 0xffff || inputs.len() > 0xff || outputs.len() > 0xff {
                bail!("row {n}: the binding does not fit its word");
            }
            words.push(14 | u64::from(*kernel) << 4 | (inputs.len() as u64) << 20 | (outputs.len() as u64) << 28 | u64::from(*direct) << 36);
            let codes: Vec<u32> = inputs.iter().chain(outputs).copied().collect();
            for pair in codes.chunks(2) {
                words.push(u64::from(pair[0]) | pair.get(1).map_or(0, |&c| u64::from(c) << 32));
            }
            continue;
        }
        let (p, a, f, x, s, k) = (width[W_P], width[W_A], width[W_F], width[W_X], width[W_S], width[W_K]);
        let (op, fields): (u32, Vec<(u32, u8)>) = match *row {
            Row::Clear { base, count } => (1, vec![(base, p), (count, p)]),
            Row::Contribution { op, init, slot, p0, p1, dst, f0, f1, factor } => {
                (op, vec![(init as u32, 1), (slot, s), (p0, p), (p1, p), (dst, p), (f0, f), (f1, f), (factor, x)])
            }
            Row::Propagator { op, base, form, factor } => (op, vec![(base, p), (form, f), (factor, x)]),
            Row::Identity { base, count, factor } => (10, vec![(base, p), (count, k), (factor, x)]),
            Row::Closure { p0, p1, amplitude, factor, first, count } => {
                (11, vec![(p0, p), (p1, p), (amplitude, a), (factor, x), (first, x), (count, k)])
            }
            Row::Source { dst, form, factor, mass, code } => (12, vec![(dst, p), (form, f), (factor, x), (mass, s), (code, 10)]),
            Row::Symjit { init, p0, p1, dst, f0, f1, factor } => {
                (13, vec![(init as u32, 1), (p0, p), (p1, p), (dst, p), (f0, f), (f1, f), (factor, x)])
            }
            Row::Binding { .. } => unreachable!(),
        };
        let (mut word, mut at) = (u64::from(op), 4u32);
        for (value, bits) in fields {
            if bits < 32 && value >> bits != 0 {
                bail!("row {n}: {value} does not fit {bits} bits");
            }
            word |= u64::from(value) << at;
            at += u32::from(bits);
        }
        if at > 64 {
            bail!("row {n}: {at} bits");
        }
        words.push(word);
    }
    Ok(words)
}
