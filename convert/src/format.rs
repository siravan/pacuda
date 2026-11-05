// SPDX-License-Identifier: 0BSD
//! The `.pacuda` file: what the C UMAMI wrapper needs to evaluate an LC artifact's contributions
//! on the recurrence runner. Little endian, sequential, no padding; every array is preceded by its
//! count, every string by its byte length (u32, no NUL).
//!
//! ```text
//! file      magic "PACUDA\0\x01", u32 version (2), u32 kernel (runner configuration: 0 df64,
//!           1 f64, 2 mixed, 3 mixed64), u32 batch (points per runner call, 0: each template's
//!           tile), u32 n, n × 64 hex bytes: the SHA-256s of
//!           the symjit kernels (an op 14 word's kernel index is a position here; the wrapper
//!           maps them to the runner's indices), u32 processes, the processes
//! process   str key, u32 legs, u32 helicities, str alpha_s parameter ("" if none), f64 its value
//!           at conversion, i32 QCD power (|M|² ∝ alpha_s^power), u32 parameters, per parameter
//!           str name, f64 real, f64 imaginary (the model parameters set at conversion: mg7's
//!           param-card values), i64 group factor numerator,
//!           i64 denominator (values · num / den when they differ), u32 crossing (0, or legs:
//!           then legs × (u32 input leg, f64 sign)), u32 colours, the colours
//! colour    str colour id, the run template, the reduction, the reference
//! template  u32 tile (RustiCol's points per plan run of this process), u32 words, u64 words[],
//!           u8 width[8], u32 current, amplitude and momentum planes,
//!           u32 scales, f64 scale_re[], scale_im[], u32 factors, f64 factor_re[], factor_im[],
//!           u32 zero runs, u32 (first, count)[], u32 download runs, u32 (first, count)[],
//!           u32 external count, u32 forms, u32 (first term, terms)[forms], u32 terms,
//!           u32 term_slots[], f64 term_coefficients[]
//! reduction u32 outputs (= helicities), u32 offsets[outputs + 1], u32 planes[offsets[outputs]],
//!           f64 weights[offsets[outputs]]   (output h: planes offsets[h] .. offsets[h+1] in order)
//! reference u32 points, f64 momenta[points × legs × 4] (point-major, the runtime's input order,
//!           before the crossing), f64 values[points × helicities] (the CPU's, group factor applied)
//! ```

use std::io::Write;

use anyhow::{anyhow, bail, Result};

pub const MAGIC: &[u8; 8] = b"PACUDA\0\x01";
pub const VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    pub tile: u32,
    pub words: Vec<u64>,
    pub width: [u8; 8],
    pub current_planes: u32,
    pub amplitude_planes: u32,
    pub momentum_planes: u32,
    pub scale_re: Vec<f64>,
    pub scale_im: Vec<f64>,
    pub factor_re: Vec<f64>,
    pub factor_im: Vec<f64>,
    pub zero_amplitude: Vec<u32>,
    pub download_amplitude: Vec<u32>,
    pub external_count: u32,
    pub form_terms: Vec<u32>,
    pub term_slots: Vec<u32>,
    pub term_coefficients: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    pub offsets: Vec<u32>,
    pub planes: Vec<u32>,
    pub weights: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Colour {
    pub id: String,
    pub template: Template,
    pub reduction: Reduction,
    pub reference_points: u32,
    pub reference_momenta: Vec<f64>,
    pub reference_values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Process {
    pub key: String,
    pub legs: u32,
    pub helicities: u32,
    pub alpha_s_name: String,
    pub alpha_s: f64,
    pub qcd_power: i32,
    /// the model parameters set before recording (name, real, imaginary)
    pub parameters: Vec<(String, f64, f64)>,
    pub factor: (i64, i64),
    pub crossing: Option<Vec<(u32, f64)>>,
    pub colours: Vec<Colour>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct File {
    pub kernel: u32,
    pub batch: u32,
    pub symjit_sha256: Vec<String>,
    pub processes: Vec<Process>,
}

struct Out(Vec<u8>);

impl Out {
    fn u8s(&mut self, v: &[u8]) {
        self.0.extend_from_slice(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend(v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend(v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend(v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend(v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.u8s(s.as_bytes());
    }
    fn u32s(&mut self, v: &[u32]) {
        v.iter().for_each(|x| self.u32(*x));
    }
    fn u64s(&mut self, v: &[u64]) {
        v.iter().for_each(|x| self.0.extend(x.to_le_bytes()));
    }
    fn f64s(&mut self, v: &[f64]) {
        v.iter().for_each(|x| self.f64(*x));
    }
    /// count + values, for an array of `(first, count)` pairs the count of pairs
    fn runs(&mut self, v: &[u32]) {
        self.u32((v.len() / 2) as u32);
        self.u32s(v);
    }
}

impl File {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Out(Vec::new());
        o.u8s(MAGIC);
        o.u32(VERSION);
        o.u32(self.kernel);
        o.u32(self.batch);
        o.u32(self.symjit_sha256.len() as u32);
        for s in &self.symjit_sha256 {
            o.u8s(s.as_bytes());
        }
        o.u32(self.processes.len() as u32);
        for p in &self.processes {
            o.str(&p.key);
            o.u32(p.legs);
            o.u32(p.helicities);
            o.str(&p.alpha_s_name);
            o.f64(p.alpha_s);
            o.i32(p.qcd_power);
            o.u32(p.parameters.len() as u32);
            for (name, re, im) in &p.parameters {
                o.str(name);
                o.f64(*re);
                o.f64(*im);
            }
            o.i64(p.factor.0);
            o.i64(p.factor.1);
            match &p.crossing {
                None => o.u32(0),
                Some(c) => {
                    o.u32(c.len() as u32);
                    for (leg, sign) in c {
                        o.u32(*leg);
                        o.f64(*sign);
                    }
                }
            }
            o.u32(p.colours.len() as u32);
            for c in &p.colours {
                o.str(&c.id);
                let t = &c.template;
                o.u32(t.tile);
                o.u32(t.words.len() as u32);
                o.u64s(&t.words);
                o.u8s(&t.width);
                o.u32(t.current_planes);
                o.u32(t.amplitude_planes);
                o.u32(t.momentum_planes);
                o.u32(t.scale_re.len() as u32);
                o.f64s(&t.scale_re);
                o.f64s(&t.scale_im);
                o.u32(t.factor_re.len() as u32);
                o.f64s(&t.factor_re);
                o.f64s(&t.factor_im);
                o.runs(&t.zero_amplitude);
                o.runs(&t.download_amplitude);
                o.u32(t.external_count);
                o.runs(&t.form_terms);
                o.u32(t.term_slots.len() as u32);
                o.u32s(&t.term_slots);
                o.f64s(&t.term_coefficients);
                let r = &c.reduction;
                o.u32((r.offsets.len() - 1) as u32);
                o.u32s(&r.offsets);
                o.u32s(&r.planes);
                o.f64s(&r.weights);
                o.u32(c.reference_points);
                o.f64s(&c.reference_momenta);
                o.f64s(&c.reference_values);
            }
        }
        o.0
    }

    pub fn write(&self, path: &std::path::Path) -> Result<()> {
        let mut f = std::fs::File::create(path).map_err(|e| anyhow!("{}: {}", path.display(), e))?;
        f.write_all(&self.to_bytes())?;
        Ok(())
    }

    pub fn read(bytes: &[u8]) -> Result<File> {
        let mut r = In { b: bytes, at: 0 };
        if r.take(8)? != MAGIC {
            bail!("not a .pacuda file");
        }
        let version = r.u32()?;
        if version != VERSION {
            bail!(".pacuda version {version}, this reader reads {VERSION}");
        }
        let (kernel, batch) = (r.u32()?, r.u32()?);
        let n = r.u32()? as usize;
        let mut symjit_sha256 = Vec::with_capacity(n);
        for _ in 0..n {
            symjit_sha256.push(String::from_utf8(r.take(64)?.to_vec())?);
        }
        let processes_n = r.u32()? as usize;
        let mut processes = Vec::with_capacity(processes_n);
        for _ in 0..processes_n {
            let key = r.str()?;
            let (legs, helicities) = (r.u32()?, r.u32()?);
            let alpha_s_name = r.str()?;
            let alpha_s = r.f64()?;
            let qcd_power = r.i32()?;
            let parameters_n = r.u32()? as usize;
            let mut parameters = Vec::with_capacity(parameters_n);
            for _ in 0..parameters_n {
                parameters.push((r.str()?, r.f64()?, r.f64()?));
            }
            let factor = (r.i64()?, r.i64()?);
            let crossing_n = r.u32()? as usize;
            let crossing = if crossing_n == 0 {
                None
            } else {
                if crossing_n != legs as usize {
                    bail!("process {key}: a crossing of {crossing_n} legs, of {legs}");
                }
                let mut c = Vec::with_capacity(crossing_n);
                for _ in 0..crossing_n {
                    c.push((r.u32()?, r.f64()?));
                }
                Some(c)
            };
            let colours_n = r.u32()? as usize;
            let mut colours = Vec::with_capacity(colours_n);
            for _ in 0..colours_n {
                let id = r.str()?;
                let tile = r.u32()?;
                let words_n = r.u32()? as usize;
                let words = r.u64s(words_n)?;
                let width: [u8; 8] = r.take(8)?.try_into().unwrap();
                let (current_planes, amplitude_planes, momentum_planes) = (r.u32()?, r.u32()?, r.u32()?);
                let scales = r.u32()? as usize;
                let (scale_re, scale_im) = (r.f64s(scales)?, r.f64s(scales)?);
                let factors = r.u32()? as usize;
                let (factor_re, factor_im) = (r.f64s(factors)?, r.f64s(factors)?);
                let zero_amplitude = r.runs()?;
                let download_amplitude = r.runs()?;
                let external_count = r.u32()?;
                let form_terms = r.runs()?;
                let terms = r.u32()? as usize;
                let (term_slots, term_coefficients) = (r.u32s(terms)?, r.f64s(terms)?);
                let outputs = r.u32()? as usize;
                let offsets = r.u32s(outputs + 1)?;
                let count = *offsets.last().unwrap() as usize;
                let (planes, weights) = (r.u32s(count)?, r.f64s(count)?);
                let reference_points = r.u32()?;
                let reference_momenta = r.f64s(reference_points as usize * legs as usize * 4)?;
                let reference_values = r.f64s(reference_points as usize * helicities as usize)?;
                colours.push(Colour {
                    id,
                    template: Template {
                        tile,
                        words,
                        width,
                        current_planes,
                        amplitude_planes,
                        momentum_planes,
                        scale_re,
                        scale_im,
                        factor_re,
                        factor_im,
                        zero_amplitude,
                        download_amplitude,
                        external_count,
                        form_terms,
                        term_slots,
                        term_coefficients,
                    },
                    reduction: Reduction { offsets, planes, weights },
                    reference_points,
                    reference_momenta,
                    reference_values,
                });
            }
            processes.push(Process { key, legs, helicities, alpha_s_name, alpha_s, qcd_power, parameters, factor, crossing, colours });
        }
        if r.at != bytes.len() {
            bail!("{} bytes after the last process", bytes.len() - r.at);
        }
        Ok(File { kernel, batch, symjit_sha256, processes })
    }
}

struct In<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> In<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| anyhow!("the file ends early"))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        (0..n).map(|_| self.u32()).collect()
    }
    fn u64s(&mut self, n: usize) -> Result<Vec<u64>> {
        (0..n).map(|_| Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))).collect()
    }
    fn f64s(&mut self, n: usize) -> Result<Vec<f64>> {
        (0..n).map(|_| self.f64()).collect()
    }
    fn runs(&mut self) -> Result<Vec<u32>> {
        let n = self.u32()? as usize;
        self.u32s(2 * n)
    }
}
