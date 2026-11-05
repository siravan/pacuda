// SPDX-License-Identifier: 0BSD
//! The parts of a generated UMAMI provider header (`provider_p0.h`, pyAmpliCol's
//! `api_templates/umami`) the converter needs: the process keys (`umami_processes`) and the
//! (process index, colour id) of every contribution (`*_contributions[]`).

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{anyhow, bail, Result};

pub struct Provider {
    pub processes: Vec<String>,
    /// (process index, colour id) of each contribution, in header order
    pub contributions: Vec<(usize, String)>,
}

/// The first `"..."` string of `line` and what follows it.
fn quoted(line: &str) -> Option<(&str, &str)> {
    let start = line.find('"')? + 1;
    let end = start + line[start..].find('"')?;
    Some((&line[start..end], &line[end + 1..]))
}

impl Provider {
    pub fn read(path: &Path) -> Result<Provider> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow!("{}: {}", path.display(), e))?;
        let mut processes = Vec::new();
        let mut contributions = Vec::new();
        #[derive(PartialEq)]
        enum In {
            Nothing,
            Processes,
            Contributions,
        }
        let mut state = In::Nothing;
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with("static const UmamiProcess umami_processes[]") {
                state = In::Processes;
                continue;
            }
            if t.starts_with("static const UmamiContribution ") && t.contains("_contributions[]") {
                state = In::Contributions;
                continue;
            }
            if t.starts_with("};") {
                state = In::Nothing;
                continue;
            }
            match state {
                In::Processes if t.starts_with('{') => {
                    let (key, _) = quoted(t).ok_or_else(|| anyhow!("{}: a process without a key: {t}", path.display()))?;
                    processes.push(key.to_string());
                }
                In::Contributions if t.starts_with('{') => {
                    let index: usize = t[1..]
                        .split(',')
                        .next()
                        .and_then(|v| v.trim().parse().ok())
                        .ok_or_else(|| anyhow!("{}: a contribution without a process index: {t}", path.display()))?;
                    let (color, _) = quoted(t).ok_or_else(|| anyhow!("{}: a contribution without a colour id: {t}", path.display()))?;
                    contributions.push((index, color.to_string()));
                }
                _ => {}
            }
        }
        if processes.is_empty() {
            bail!("{}: no processes", path.display());
        }
        if let Some((index, _)) = contributions.iter().find(|(i, _)| *i >= processes.len()) {
            bail!("{}: a contribution of process {index}, of {}", path.display(), processes.len());
        }
        Ok(Provider { processes, contributions })
    }

    /// The distinct (process key, colour id) pairs of the contributions, sorted.
    pub fn used_pairs(&self) -> Vec<(String, String)> {
        let set: BTreeSet<(String, String)> = self
            .contributions
            .iter()
            .map(|(i, c)| (self.processes[*i].clone(), c.clone()))
            .collect();
        set.into_iter().collect()
    }
}
