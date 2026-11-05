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

use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use pacuda_convert::record::*;
use pacuda_convert::{format, provider, replay};

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
                let file = convert_artifact(lc, points, &parameters, kernel.unwrap_or(0), batch, 8)?.0;
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
