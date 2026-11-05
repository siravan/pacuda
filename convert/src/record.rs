// SPDX-License-Identifier: 0BSD
//! The recording of a pyAmpliCol LC artifact's recurrence plan runs (the patched RustiCol's record
//! mode), shared by `pacuda-convert` and `pac2json`.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use rusticol_core::recurrence::direct_backend::{pacman_record, PacmanRecording};
use rusticol_core::NativeRuntime;

use crate::{format, replay};

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
    /// the runtime's ids of the helicity outputs, in output order
    pub helicity_ids: Vec<String>,
    pub crossing: Option<Vec<(usize, f64)>>,
    pub group_factor: (i64, i64),
    pub momenta: Vec<f64>,
    pub values: Vec<f64>,
    pub recording: PacmanRecording,
}

/// The runtime of `process` with `parameters` set (those it has), and the ones set.
pub fn load(lc: &Path, process: &str, parameters: &[(String, f64)]) -> Result<(NativeRuntime, Vec<(String, f64, f64)>)> {
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
        helicity_ids: resolved.helicity_ids.clone(),
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
pub fn colour_of(id: &str, r: &Recorded, kernels: &mut Vec<String>, reference: usize) -> Result<format::Colour> {
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

pub fn process_of(key: &str, r: &Recorded, alpha_s_name: &str, alpha_s: f64, qcd_power: i32, parameters: Vec<(String, f64, f64)>) -> format::Process {
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
pub fn alpha_s_power(lc: &Path, process: &str, name: &str, parameters: &[(String, f64)]) -> Result<(f64, i32)> {
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
pub fn extract_kernels(lc: &Path, sha256: &[String], dir: &Path) -> Result<usize> {
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
pub fn parse_parameters(text: &str) -> Result<Vec<(String, f64)>> {
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

pub fn kernel_number(name: Option<&String>) -> Result<u32> {
    Ok(match name.map(String::as_str) {
        Some("df64") => 0,
        Some("f64") => 1,
        Some("mixed") => 2,
        Some("mixed64") => 3,
        other => bail!("--kernel {other:?}: df64, f64, mixed or mixed64"),
    })
}


/// The `.pacuda` file of the LC artifact `lc`: for each (process, colour id) its provider uses, a
/// checked recording of `points` points, `reference` reference points; and per process of the file
/// the ids of its helicity outputs.
pub fn convert_artifact(
    lc: &Path,
    points: usize,
    parameters: &[(String, f64)],
    kernel: u32,
    batch: u32,
    reference: usize,
) -> Result<(format::File, Vec<Vec<String>>)> {
    let provider = crate::provider::Provider::read(&lc.join("API/umami/provider_p0.h"))?;
    let mut file = format::File { kernel, batch, symjit_sha256: Vec::new(), processes: Vec::new() };
    let mut helicity_ids = Vec::new();
    let pairs = provider.used_pairs();
    for (index, key) in provider.processes.iter().enumerate() {
        let colours: Vec<&String> = pairs.iter().filter(|(p, _)| p == key).map(|(_, c)| c).collect();
        if colours.is_empty() {
            continue;
        }
        let (_, set) = load(lc, key, parameters)?;
        let (alpha_s_name, alpha_s, qcd_power) = match &provider.alpha_s[index] {
            Some(name) => {
                let (a, p) = alpha_s_power(lc, key, name, parameters)?;
                (name.clone(), a, p)
            }
            None => (String::new(), 0.0, 0),
        };
        let mut process: Option<format::Process> = None;
        let mut ids: Option<Vec<String>> = None;
        for color in colours {
            let r = record_one(lc, key, color, points, parameters)?;
            let problems = check_recording(&r, points);
            if !problems.is_empty() {
                bail!("{key} {color}: {}", problems.join("; "));
            }
            match &ids {
                None => ids = Some(r.helicity_ids.clone()),
                Some(ids) if *ids != r.helicity_ids => bail!("{key} {color}: other helicity outputs than its first colour's"),
                Some(_) => {}
            }
            let c = colour_of(color, &r, &mut file.symjit_sha256, reference)?;
            process
                .get_or_insert_with(|| process_of(key, &r, &alpha_s_name, alpha_s, qcd_power, set.clone()))
                .colours
                .push(c);
        }
        if let Some(p) = process {
            file.processes.push(p);
            helicity_ids.push(ids.unwrap_or_default());
        }
    }
    Ok((file, helicity_ids))
}
