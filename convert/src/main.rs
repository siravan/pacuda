// SPDX-License-Identifier: 0BSD
//! pacuda-convert: records the recurrence plan runs of a pyAmpliCol LC artifact, as an mg7 UMAMI
//! library evaluates them, with the record mode of the patched RustiCol (no GPU needed).
//!
//!   pacuda-convert record LC_DIR [-n POINTS] [--replay librecurrence.so [--kernel K]]
//!
//! `record`: for each (process, colour id) the artifact's UMAMI provider (`API/umami/provider_p0.h`)
//! uses, one resolved evaluation of POINTS random points (default 64) under recording, then a
//! summary and checks (no errors; one plan run per tile; every run reduced; every computed
//! helicity output reduced into). With `--replay`, each recording is also replayed on that runner
//! (`replay.rs`; K = f64 by default) and compared with the CPU's values.

mod provider;
mod replay;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use rusticol_core::recurrence::direct_backend::{pacman_record, PacmanRecording};
use rusticol_core::NativeRuntime;

/// `points` sets of `n` massless momenta (two incoming along z, `n - 2` outgoing), point-major
/// `[point][leg][E, px, py, pz]`, from a fixed seed. Any finite momenta do for recording; these
/// give the reference values a physical meaning.
pub fn momenta(n: usize, points: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let ecm = 1000.0f64;
    let mut out = Vec::with_capacity(points * n * 4);
    for _ in 0..points {
        // RAMBO (massless)
        let m = n - 2;
        let mut q = vec![[0.0f64; 4]; m];
        for p in q.iter_mut() {
            let c = 2.0 * uniform() - 1.0;
            let s = (1.0 - c * c).sqrt();
            let f = 2.0 * std::f64::consts::PI * uniform();
            let e = -(uniform() * uniform()).ln();
            *p = [e, e * s * f.cos(), e * s * f.sin(), e * c];
        }
        let total: [f64; 4] = (0..4).map(|k| q.iter().map(|p| p[k]).sum()).collect::<Vec<_>>().try_into().unwrap();
        let mass = (total[0] * total[0] - total[1] * total[1] - total[2] * total[2] - total[3] * total[3]).sqrt();
        let b = [-total[1] / mass, -total[2] / mass, -total[3] / mass];
        let (x, g) = (ecm / mass, total[0] / mass);
        let a = 1.0 / (1.0 + g);
        out.extend([ecm / 2.0, 0.0, 0.0, ecm / 2.0, ecm / 2.0, 0.0, 0.0, -ecm / 2.0]);
        for p in &q {
            let bq = b[0] * p[1] + b[1] * p[2] + b[2] * p[3];
            out.push(x * (g * p[0] + bq));
            for k in 0..3 {
                out.push(x * (p[k + 1] + b[k] * (p[0] + a * bq)));
            }
        }
    }
    out
}

/// One recorded (process, colour id) and the CPU's values.
pub struct Recorded {
    pub process: String,
    pub color: String,
    pub external_count: usize,
    pub helicity_count: usize,
    pub crossing: Option<Vec<(usize, f64)>>,
    pub group_factor: (i64, i64),
    pub momenta: Vec<f64>,
    pub values: Vec<f64>,
    pub recording: PacmanRecording,
}

pub fn record_one(lc: &Path, process: &str, color: &str, points: usize) -> Result<Recorded> {
    let mut runtime = NativeRuntime::load(lc, Some(process), None)
        .map_err(|e| anyhow!("{}: loading {}: {}", lc.display(), process, e))?;
    let n = runtime.external_count();
    let momenta = momenta(n, points, 0x5eed_0000 + points as u64);
    let colors = vec![color.to_string()];
    let (result, recording) =
        pacman_record(|| runtime.evaluate_resolved_f64(&momenta, points, None, Some(&colors)));
    let resolved = result.map_err(|e| anyhow!("{} {}: evaluation: {}", process, color, e))?;
    let factor = runtime.process_group_factor();
    Ok(Recorded {
        process: process.to_string(),
        color: color.to_string(),
        external_count: n,
        helicity_count: resolved.helicity_ids.len(),
        crossing: runtime.pacman_input_crossing(),
        group_factor: (factor.numerator as i64, factor.denominator as i64),
        momenta,
        values: resolved.values,
        recording,
    })
}

/// The checks a recording must pass to be replayed: no errors, every run reduced, the tiles
/// cover the points, and the helicity outputs the values have (nonzero ones) are reduced into.
pub fn check(r: &Recorded, points: usize) -> Vec<String> {
    let mut problems: Vec<String> = r.recording.errors.clone();
    let rec = &r.recording;
    if rec.runs.is_empty() {
        problems.push("no plan run was recorded".into());
    }
    let covered: u64 = rec.runs.iter().map(|run| u64::from(run.points)).sum();
    if covered != points as u64 {
        problems.push(format!("the runs cover {covered} points, the evaluation {points}"));
    }
    for (k, run) in rec.runs.iter().enumerate() {
        if !run.upload_amplitude.is_empty() {
            problems.push(format!("run {k} uploads amplitudes"));
        }
        if run.reductions.is_empty() {
            problems.push(format!("run {k} has no reduction"));
        }
        if run.reductions.iter().any(|red| red.color != 0) {
            problems.push(format!("run {k} reduces into another colour output"));
        }
    }
    let reduced: BTreeSet<u32> = rec.runs.iter().flat_map(|run| run.reductions.iter().map(|red| red.helicity)).collect();
    let h = r.helicity_count;
    for hel in 0..h {
        let nonzero = (0..points).any(|p| r.values[p * h + hel] != 0.0);
        if nonzero && !reduced.contains(&(hel as u32)) {
            problems.push(format!("helicity output {hel} has values but no reduction"));
        }
    }
    problems
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = (|| -> Result<()> {
        let command = args.first().ok_or_else(|| anyhow!("usage: pacuda-convert record LC_DIR [-n POINTS]"))?;
        let (mut paths, mut points) = (Vec::<PathBuf>::new(), 64usize);
        let (mut runner_path, mut kernel) = (None::<PathBuf>, 1u32);
        let mut it = args[1..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "-n" => points = it.next().ok_or_else(|| anyhow!("-n needs a value"))?.parse()?,
                "--replay" => runner_path = it.next().map(PathBuf::from),
                "--kernel" => {
                    kernel = match it.next().map(String::as_str) {
                        Some("df64") => 0,
                        Some("f64") => 1,
                        Some("mixed") => 2,
                        Some("mixed64") => 3,
                        other => bail!("--kernel {other:?}: f64, df64, mixed or mixed64"),
                    }
                }
                _ => paths.push(PathBuf::from(a)),
            }
        }
        match command.as_str() {
            "record" => {
                let lc = paths.first().ok_or_else(|| anyhow!("record needs LC_DIR"))?;
                let provider = provider::Provider::read(&lc.join("API/umami/provider_p0.h"))?;
                let mut failed = 0;
                let runner = runner_path.as_deref().map(|p| replay::Runner::load(p, kernel)).transpose()?;
                for (process, color) in provider.used_pairs() {
                    let r = record_one(lc, &process, &color, points)?;
                    let mut problems = check(&r, points);
                    let mut compared = String::new();
                    if let (Some(runner), true) = (&runner, problems.is_empty()) {
                        match replay::replay(runner, &r) {
                            Ok(values) => {
                                let differ = values.iter().zip(&r.values).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                                let worst = values
                                    .iter()
                                    .zip(&r.values)
                                    .map(|(a, b)| if *b != 0.0 { (a / b - 1.0).abs() } else { a.abs() })
                                    .fold(0.0f64, f64::max);
                                compared = format!(", replay: {differ} of {} values differ (max rel {worst:.1e})", values.len());
                                if kernel == 1 && differ > 0 {
                                    problems.push("the f64 replay is not bit-identical".into());
                                }
                            }
                            Err(e) => problems.push(format!("replay: {e:#}")),
                        }
                    }
                    let rec = &r.recording;
                    let words: usize = rec.runs.iter().map(|run| run.words.len()).sum();
                    let reductions: usize = rec.runs.iter().map(|run| run.reductions.len()).sum();
                    println!(
                        "{process} {color}: {} legs, {} helicities, {} runs (points {:?}), {} words, {} reductions, {} symjit kernels, crossing {}, factor {}/{}{compared}{}",
                        r.external_count,
                        r.helicity_count,
                        rec.runs.len(),
                        rec.runs.iter().map(|run| run.points).collect::<Vec<_>>(),
                        words,
                        reductions,
                        rec.symjit_sha256.len(),
                        r.crossing.as_ref().map_or("identity".into(), |c| format!("{c:?}")),
                        r.group_factor.0,
                        r.group_factor.1,
                        if problems.is_empty() { String::new() } else { format!("\n  PROBLEMS: {}", problems.join("; ")) }
                    );
                    failed += usize::from(!problems.is_empty());
                }
                if failed > 0 {
                    bail!("{failed} recordings have problems");
                }
                Ok(())
            }
            other => bail!("unknown command `{other}`"),
        }
    })();
    if let Err(e) = result {
        eprintln!("pacuda-convert: {e:#}");
        std::process::exit(1);
    }
}
