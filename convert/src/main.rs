// SPDX-License-Identifier: 0BSD
//! pacuda-convert: turns a pyAmpliCol LC artifact into a `.pacuda` file, by recording its
//! recurrence plan runs as an mg7 UMAMI library evaluates them, with the record mode of the
//! patched RustiCol (no GPU needed).
//!
//!   pacuda-convert convert LC_DIR -o OUT.pacuda [--parameters FILE.json] [--kernels DIR] [--kernel K] [--batch N] [-n POINTS]
//!   pacuda-convert check FILE.pacuda --runner librecurrence.so [--kernel K] [--tile T] [--host-reduce]
//!   pacuda-convert record LC_DIR [-n POINTS] [--replay librecurrence.so] [--kernel K] [--host-reduce]
//!
//! `--parameters FILE.json` ({"name": value}): the model parameters mg7 sets on the library (its
//! param card's values, `amplicol_parameters`), set on every runtime before recording; the
//! file records them (the wrapper accepts only these values).
//! `convert`: for each (process, colour id) the artifact's UMAMI provider (`API/umami/provider_p0.h`)
//! uses, one resolved evaluation of POINTS random points (default 1000) under recording; the
//! checks (`check_recording`); the run template (identical across the recorded tiles), the
//! grouped reduction and 8 reference points; per process the alpha_s power, measured on the CPU
//! (2 alpha_s0 against alpha_s0, an exact power of 2 required). `--kernels DIR` writes the symjit
//! kernels the file needs (`<sha256>.plane.symjit`, for `pacuda-gen cuda`). K (the runner
//! configuration the wrapper uses): df64 (default), f64, mixed, mixed64; N: points per runner
//! call (0: RustiCol's tile).
//! `check`: evaluates every colour's reference points from the file on the runner (K = f64 by
//! default; in tiles of T, default the file's tile) and compares them with the CPU's values.
//! `record`: the recordings and their checks only, optionally evaluated on a runner.

mod format;
mod provider;
mod replay;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use rusticol_core::recurrence::direct_backend::{pacman_record, PacmanRecording};
use rusticol_core::NativeRuntime;

/// `points` sets of `n` massless momenta (two incoming along z, `n - 2` outgoing by RAMBO),
/// point-major `[point][leg][E, px, py, pz]`, from a fixed seed. Any finite momenta do for
/// recording; these give the reference values a physical meaning.
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
        let m = n - 2;
        let mut q = vec![[0.0f64; 4]; m];
        for p in q.iter_mut() {
            let c = 2.0 * uniform() - 1.0;
            let s = (1.0 - c * c).sqrt();
            let f = 2.0 * std::f64::consts::PI * uniform();
            let e = -(uniform() * uniform()).ln();
            *p = [e, e * s * f.cos(), e * s * f.sin(), e * c];
        }
        let total: [f64; 4] = std::array::from_fn(|k| q.iter().map(|p| p[k]).sum());
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
    pub external_count: usize,
    pub helicity_count: usize,
    pub crossing: Option<Vec<(usize, f64)>>,
    pub group_factor: (i64, i64),
    pub momenta: Vec<f64>,
    pub values: Vec<f64>,
    pub recording: PacmanRecording,
}

/// The runtime of `process` with `parameters` set (those it has), and the ones set.
fn load(lc: &Path, process: &str, parameters: &[(String, f64)]) -> Result<(NativeRuntime, Vec<(String, f64, f64)>)> {
    let mut runtime =
        NativeRuntime::load(lc, Some(process), None).map_err(|e| anyhow!("{}: loading {}: {}", lc.display(), process, e))?;
    let mut set = Vec::new();
    for (name, value) in parameters {
        if runtime.model_parameter_value(name).is_err() {
            continue;
        }
        runtime
            .set_model_parameter(name, *value, 0.0)
            .map_err(|e| anyhow!("{process}: setting {name} = {value}: {e}"))?;
        set.push((name.clone(), *value, 0.0));
    }
    Ok((runtime, set))
}

pub fn record_one(lc: &Path, process: &str, color: &str, points: usize, parameters: &[(String, f64)]) -> Result<Recorded> {
    let (mut runtime, _) = load(lc, process, parameters)?;
    let n = runtime.external_count();
    let momenta = momenta(n, points, 0x5eed_0000 + points as u64);
    let colors = vec![color.to_string()];
    let (result, recording) =
        pacman_record(|| runtime.evaluate_resolved_f64(&momenta, points, None, Some(&colors)));
    let resolved = result.map_err(|e| anyhow!("{} {}: evaluation: {}", process, color, e))?;
    let factor = runtime.process_group_factor();
    Ok(Recorded {
        external_count: n,
        helicity_count: resolved.helicity_ids.len(),
        crossing: runtime.pacman_input_crossing(),
        group_factor: (factor.numerator as i64, factor.denominator as i64),
        momenta,
        values: resolved.values,
        recording,
    })
}

/// The checks a recording must pass: no errors, the tiles cover the points, no amplitude uploads,
/// every run reduced, one colour output, every helicity output with values reduced into, and
/// every run's static part (all but its points) the first run's.
pub fn check_recording(r: &Recorded, points: usize) -> Vec<String> {
    let mut problems: Vec<String> = r.recording.errors.clone();
    let rec = &r.recording;
    if rec.runs.is_empty() {
        problems.push("no plan run was recorded".into());
        return problems;
    }
    let covered: u64 = rec.runs.iter().map(|run| u64::from(run.points)).sum();
    if covered != points as u64 {
        problems.push(format!("the runs cover {covered} points, the evaluation {points}"));
    }
    let first = &rec.runs[0];
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
        let same = run.words == first.words
            && run.width == first.width
            && (run.current_planes, run.amplitude_planes, run.momentum_planes)
                == (first.current_planes, first.amplitude_planes, first.momentum_planes)
            && run.scale_re == first.scale_re
            && run.scale_im == first.scale_im
            && run.factor_re == first.factor_re
            && run.factor_im == first.factor_im
            && run.zero_amplitude == first.zero_amplitude
            && run.download_amplitude == first.download_amplitude
            && run.external_count == first.external_count
            && run.form_terms == first.form_terms
            && run.term_slots == first.term_slots
            && run.term_coefficients == first.term_coefficients
            && run.reductions == first.reductions
            && run.stride == first.stride;
        if !same {
            problems.push(format!("run {k}'s static part differs from run 0's"));
        }
    }
    let reduced: BTreeSet<u32> = first.reductions.iter().map(|red| red.helicity).collect();
    let h = r.helicity_count;
    for hel in 0..h {
        let nonzero = (0..points).any(|p| r.values[p * h + hel] != 0.0);
        if nonzero && !reduced.contains(&(hel as u32)) {
            problems.push(format!("helicity output {hel} has values but no reduction"));
        }
    }
    problems
}

/// The file's colour of a checked recording: its first run as the template (op 14 kernel indices
/// mapped into the file's table `kernels`, extended as needed), the grouped reduction, and the
/// first `reference` points.
fn colour_of(id: &str, r: &Recorded, kernels: &mut Vec<String>, reference: usize) -> Result<format::Colour> {
    let map: Vec<u32> = r
        .recording
        .symjit_sha256
        .iter()
        .map(|s| {
            let at = kernels.iter().position(|k| k == s).unwrap_or_else(|| {
                kernels.push(s.clone());
                kernels.len() - 1
            });
            at as u32
        })
        .collect();
    let run = &r.recording.runs[0];
    let h = r.helicity_count;
    let (mut planes, mut weights, mut offsets) = (Vec::new(), Vec::new(), vec![0u32]);
    for hel in 0..h as u32 {
        for red in run.reductions.iter().filter(|red| red.helicity == hel) {
            planes.push(red.plane);
            weights.push(red.weight);
        }
        offsets.push(planes.len() as u32);
    }
    let n = r.external_count;
    let reference = reference.min(r.values.len() / h.max(1));
    Ok(format::Colour {
            id: id.to_string(),
            template: format::Template {
                tile: run.stride,
                words: replay::remap_kernels(&run.words, &map)?,
                width: run.width,
                current_planes: run.current_planes,
                amplitude_planes: run.amplitude_planes,
                momentum_planes: run.momentum_planes,
                scale_re: run.scale_re.clone(),
                scale_im: run.scale_im.clone(),
                factor_re: run.factor_re.clone(),
                factor_im: run.factor_im.clone(),
                zero_amplitude: run.zero_amplitude.clone(),
                download_amplitude: run.download_amplitude.clone(),
                external_count: run.external_count,
                form_terms: run.form_terms.clone(),
                term_slots: run.term_slots.clone(),
                term_coefficients: run.term_coefficients.clone(),
            },
            reduction: format::Reduction { offsets, planes, weights },
            reference_points: reference as u32,
            reference_momenta: r.momenta[..reference * n * 4].to_vec(),
            reference_values: r.values[..reference * h].to_vec(),
        })
}

fn process_of(key: &str, r: &Recorded, alpha_s_name: &str, alpha_s: f64, qcd_power: i32, parameters: Vec<(String, f64, f64)>) -> format::Process {
    format::Process {
        key: key.to_string(),
        legs: r.external_count as u32,
        helicities: r.helicity_count as u32,
        alpha_s_name: alpha_s_name.to_string(),
        alpha_s,
        qcd_power,
        parameters,
        factor: r.group_factor,
        crossing: r.crossing.as_ref().map(|c| c.iter().map(|&(l, s)| (l as u32, s)).collect()),
        colours: Vec::new(),
    }
}

/// alpha_s0 and the power p of |M|² ∝ alpha_s^p, from the CPU's totals at alpha_s0 and 2 alpha_s0
/// (every point's ratio must be 2^p to 1e-12; else the process is refused).
fn alpha_s_power(lc: &Path, process: &str, name: &str, parameters: &[(String, f64)]) -> Result<(f64, i32)> {
    let (mut runtime, _) = load(lc, process, parameters)?;
    let (alpha, imaginary) = runtime.model_parameter_value(name).map_err(|e| anyhow!("{process}: {name}: {e}"))?;
    if imaginary != 0.0 || !(alpha > 0.0) {
        bail!("{process}: {name} = ({alpha}, {imaginary})");
    }
    let n = runtime.external_count();
    let points = 16;
    let momenta = momenta(n, points, 0xa1fa);
    let totals = |runtime: &mut NativeRuntime| -> Result<Vec<f64>> {
        let r = runtime.evaluate_resolved_f64(&momenta, points, None, None).map_err(|e| anyhow!("{process}: {e}"))?;
        let per = r.values.len() / points;
        Ok(r.values.chunks(per).map(|c| c.iter().sum()).collect())
    };
    let base = totals(&mut runtime)?;
    runtime.set_model_parameter(name, 2.0 * alpha, 0.0).map_err(|e| anyhow!("{process}: {e}"))?;
    let doubled = totals(&mut runtime)?;
    let ratios: Vec<f64> = base.iter().zip(&doubled).filter(|(b, _)| **b != 0.0).map(|(b, d)| d / b).collect();
    let first = *ratios.first().ok_or_else(|| anyhow!("{process}: every total is zero"))?;
    let power = first.log2().round() as i32;
    let want = 2f64.powi(power);
    if let Some(bad) = ratios.iter().find(|r| (*r / want - 1.0).abs() > 1e-12) {
        bail!("{process}: |M|² is not a power of alpha_s (ratio {bad} at 2 alpha_s, 2^{power} = {want})");
    }
    Ok((alpha, power))
}

/// Writes the `.plane.symjit` members of `lc/evaluators.pacbin` with these SHA-256s into `dir`.
fn extract_kernels(lc: &Path, sha256: &[String], dir: &Path) -> Result<usize> {
    std::fs::create_dir_all(dir)?;
    let pack = rusticol_core::pacbin::PacbinReader::open(lc.join("evaluators.pacbin")).map_err(|e| anyhow!("{}: {e}", lc.display()))?;
    let mut found = BTreeSet::new();
    for member in pack.members() {
        let hex: String = member.sha256().iter().map(|b| format!("{b:02x}")).collect();
        if member.logical_path().ends_with(".plane.symjit") && sha256.contains(&hex) && found.insert(hex.clone()) {
            let bytes = pack.member_bytes(member.logical_path()).map_err(|e| anyhow!("{e}"))?;
            std::fs::write(dir.join(format!("{hex}.plane.symjit")), bytes)?;
        }
    }
    if let Some(missing) = sha256.iter().find(|s| !found.contains(*s)) {
        bail!("{}: no plane application with SHA-256 {missing}", lc.display());
    }
    Ok(found.len())
}

/// A flat JSON object of numbers, {"name": value, ...} (what pacuda.py writes).
fn parse_parameters(text: &str) -> Result<Vec<(String, f64)>> {
    let body = text.trim().strip_prefix('{').and_then(|t| t.strip_suffix('}')).ok_or_else(|| anyhow!("not a JSON object"))?;
    let mut out = Vec::new();
    for item in body.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, value) = item.split_once(':').ok_or_else(|| anyhow!("bad entry {item}"))?;
        let name = name.trim().trim_matches('"').to_string();
        let value: f64 = value.trim().parse().map_err(|_| anyhow!("{name}: not a number: {value}"))?;
        out.push((name, value));
    }
    Ok(out)
}

fn kernel_number(name: Option<&String>) -> Result<u32> {
    Ok(match name.map(String::as_str) {
        Some("df64") => 0,
        Some("f64") => 1,
        Some("mixed") => 2,
        Some("mixed64") => 3,
        other => bail!("--kernel {other:?}: df64, f64, mixed or mixed64"),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = (|| -> Result<()> {
        let command = args.first().ok_or_else(|| anyhow!("usage: pacuda-convert convert|check|record ..."))?;
        let mut paths = Vec::<PathBuf>::new();
        let (mut points, mut out, mut kernels_dir, mut runner_path) = (None::<usize>, None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
        let mut parameters: Vec<(String, f64)> = Vec::new();
        let (mut kernel, mut batch, mut tile, mut host_reduce) = (None::<u32>, 0u32, None::<usize>, false);
        let mut it = args[1..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "-n" => points = Some(it.next().ok_or_else(|| anyhow!("-n needs a value"))?.parse()?),
                "-o" => out = it.next().map(PathBuf::from),
                "--kernels" => kernels_dir = it.next().map(PathBuf::from),
                "--parameters" => {
                    let path = it.next().ok_or_else(|| anyhow!("--parameters needs a file"))?;
                    parameters = parse_parameters(&std::fs::read_to_string(path).map_err(|e| anyhow!("{path}: {e}"))?)
                        .map_err(|e| anyhow!("{path}: {e}"))?;
                }
                "--runner" | "--replay" => runner_path = it.next().map(PathBuf::from),
                "--kernel" => kernel = Some(kernel_number(it.next())?),
                "--batch" => batch = it.next().ok_or_else(|| anyhow!("--batch needs a value"))?.parse()?,
                "--tile" => tile = Some(it.next().ok_or_else(|| anyhow!("--tile needs a value"))?.parse()?),
                "--host-reduce" => host_reduce = true,
                _ => paths.push(PathBuf::from(a)),
            }
        }
        match command.as_str() {
            "convert" => {
                let lc = paths.first().ok_or_else(|| anyhow!("convert needs LC_DIR"))?;
                let out = out.ok_or_else(|| anyhow!("convert needs -o OUT.pacuda"))?;
                let points = points.unwrap_or(1000);
                let provider = provider::Provider::read(&lc.join("API/umami/provider_p0.h"))?;
                let mut file = format::File {
                    kernel: kernel.unwrap_or(0),
                    batch,
                    symjit_sha256: Vec::new(),
                    processes: Vec::new(),
                };
                let pairs = provider.used_pairs();
                for (index, key) in provider.processes.iter().enumerate() {
                    let colours: Vec<&String> = pairs.iter().filter(|(p, _)| p == key).map(|(_, c)| c).collect();
                    if colours.is_empty() {
                        continue;
                    }
                    let (_, set) = load(lc, key, &parameters)?;
                    let (alpha_s_name, alpha_s, qcd_power) = match &provider.alpha_s[index] {
                        Some(name) => {
                            let (a, p) = alpha_s_power(lc, key, name, &parameters)?;
                            (name.clone(), a, p)
                        }
                        None => (String::new(), 0.0, 0),
                    };
                    let mut process: Option<format::Process> = None;
                    for color in colours {
                        let r = record_one(lc, key, color, points, &parameters)?;
                        let problems = check_recording(&r, points);
                        if !problems.is_empty() {
                            bail!("{key} {color}: {}", problems.join("; "));
                        }
                        let c = colour_of(color, &r, &mut file.symjit_sha256, 8)?;
                        process
                            .get_or_insert_with(|| process_of(key, &r, &alpha_s_name, alpha_s, qcd_power, set.clone()))
                            .colours
                            .push(c);
                    }
                    file.processes.extend(process);
                }
                file.write(&out)?;
                let colours: usize = file.processes.iter().map(|p| p.colours.len()).sum();
                println!(
                    "{}: {} processes, {} colours, {} symjit kernels, {} bytes",
                    out.display(),
                    file.processes.len(),
                    colours,
                    file.symjit_sha256.len(),
                    std::fs::metadata(&out)?.len()
                );
                for p in &file.processes {
                    println!(
                        "  {}: {} legs, {} helicities, {} colours (tile {}), {} = {} (power {}), factor {}/{}, crossing {}",
                        p.key,
                        p.legs,
                        p.helicities,
                        p.colours.len(),
                        p.colours[0].template.tile,
                        if p.alpha_s_name.is_empty() { "no alpha_s" } else { &p.alpha_s_name },
                        p.alpha_s,
                        p.qcd_power,
                        p.factor.0,
                        p.factor.1,
                        if p.crossing.is_some() { "yes" } else { "identity" }
                    );
                }
                if let Some(dir) = kernels_dir {
                    let n = extract_kernels(lc, &file.symjit_sha256, &dir)?;
                    println!("{n} symjit kernels in {}", dir.display());
                }
                // the file reads back as written
                if format::File::read(&std::fs::read(&out)?)? != file {
                    bail!("{}: the file does not read back as written", out.display());
                }
                Ok(())
            }
            "check" => {
                let path = paths.first().ok_or_else(|| anyhow!("check needs FILE.pacuda"))?;
                let file = format::File::read(&std::fs::read(path)?)?;
                let runner_path = runner_path.ok_or_else(|| anyhow!("check needs --runner librecurrence.so"))?;
                let kernel = kernel.unwrap_or(1);
                let runner = replay::Runner::load(&runner_path, kernel)?;
                let map = runner.kernel_map(&file.symjit_sha256)?;
                let mut failed = 0;
                for p in &file.processes {
                    for c in &p.colours {
                        let tile = tile.unwrap_or(c.template.tile as usize);
                        let got = replay::evaluate(&runner, &map, p, c, &c.reference_momenta, c.reference_points as usize, tile, host_reduce)?;
                        let (differ, worst, worst_total) = replay::compare(&got, &c.reference_values, p.helicities as usize);
                        println!(
                            "{} {}: {differ} of {} values differ (max {worst:.1e} of the point's total; total max rel {worst_total:.1e})",
                            p.key,
                            c.id,
                            got.len()
                        );
                        failed += usize::from(kernel == 1 && differ > 0);
                    }
                }
                if failed > 0 {
                    bail!("{failed} colours are not bit-identical in f64");
                }
                Ok(())
            }
            "record" => {
                let lc = paths.first().ok_or_else(|| anyhow!("record needs LC_DIR"))?;
                let points = points.unwrap_or(64);
                let provider = provider::Provider::read(&lc.join("API/umami/provider_p0.h"))?;
                let kernel = kernel.unwrap_or(1);
                let runner = runner_path.as_deref().map(|p| replay::Runner::load(p, kernel)).transpose()?;
                let mut failed = 0;
                for (process, color) in provider.used_pairs() {
                    let r = record_one(lc, &process, &color, points, &parameters)?;
                    let mut problems = check_recording(&r, points);
                    let mut compared = String::new();
                    if let (Some(runner), true) = (&runner, problems.is_empty()) {
                        let mut kernels = Vec::new();
                        let c = colour_of(&color, &r, &mut kernels, points)?;
                        let stride = c.template.tile;
                        let p = process_of(&process, &r, "", 0.0, 0, Vec::new());
                        let map = runner.kernel_map(&kernels)?;
                        match replay::evaluate(runner, &map, &p, &c, &r.momenta, points, stride as usize, host_reduce) {
                            Ok(values) => {
                                let (differ, worst, worst_total) = replay::compare(&values, &r.values, r.helicity_count);
                                compared = format!(
                                    ", replay: {differ} of {} values differ (max {worst:.1e} of the point's total; total max rel {worst_total:.1e})",
                                    values.len()
                                );
                                if kernel == 1 && differ > 0 {
                                    problems.push("the f64 replay is not bit-identical".into());
                                }
                            }
                            Err(e) => problems.push(format!("replay: {e:#}")),
                        }
                    }
                    let rec = &r.recording;
                    println!(
                        "{process} {color}: {} legs, {} helicities, {} runs (points {:?}), {} words, {} reductions, {} symjit kernels{compared}{}",
                        r.external_count,
                        r.helicity_count,
                        rec.runs.len(),
                        rec.runs.iter().map(|run| run.points).collect::<Vec<_>>(),
                        rec.runs.iter().map(|run| run.words.len()).sum::<usize>(),
                        rec.runs.iter().map(|run| run.reductions.len()).sum::<usize>(),
                        rec.symjit_sha256.len(),
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
