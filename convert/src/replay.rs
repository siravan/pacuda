// SPDX-License-Identifier: 0BSD
//! Replays a recording on a recurrence runner (`librecurrence.so`) and compares the values with the
//! CPU's: the check that a recording holds everything an evaluation needs. Each recorded run is
//! one runner call (its symjit kernel indices mapped to the runner's); its downloaded amplitudes
//! are reduced on the host in the recorded order (`accumulate_lc_diagonal_amplitude`), then the
//! process-group factor is applied (`apply_process_group_factor`).

use anyhow::{anyhow, bail, Result};

use crate::Recorded;

/// `pacman_recurrence_call` of `runner/pacman_recurrence.h`.
#[repr(C)]
struct RunnerCall {
    abi_version: u32,
    struct_size: u32,
    words: *const u64,
    words_key: u64,
    word_count: u32,
    width: [u8; 8],
    current_re: *mut f64,
    current_im: *mut f64,
    amplitude_re: *mut f64,
    amplitude_im: *mut f64,
    momenta: *const f64,
    scale_re: *const f64,
    scale_im: *const f64,
    factor_re: *const f64,
    factor_im: *const f64,
    upload_current: *const u32,
    upload_amplitude: *const u32,
    download_amplitude: *const u32,
    download_current: *const u32,
    current_planes: u32,
    amplitude_planes: u32,
    momentum_planes: u32,
    scale_count: u32,
    factor_count: u32,
    upload_current_runs: u32,
    upload_amplitude_runs: u32,
    download_amplitude_runs: u32,
    download_current_runs: u32,
    stride: u32,
    points: u32,
    zero_amplitude: *const u32,
    zero_amplitude_runs: u32,
    flags: u32,
    external_momenta: *const f64,
    external_count: u32,
    term_count: u32,
    form_terms: *const u32,
    term_slots: *const u32,
    term_coefficients: *const f64,
}

/// `pacman_recurrence_options`.
#[repr(C)]
struct Options {
    abi_version: u32,
    struct_size: u32,
    kernel: u32,
    block: u32,
    no_sort: u32,
    stats: u32,
}

const FORMS_ON_DEVICE: u32 = 2;

pub struct Runner {
    run: unsafe extern "C" fn(*const RunnerCall) -> i32,
    symjit_kernel: unsafe extern "C" fn(*const std::ffi::c_char) -> i32,
    _library: libloading::Library,
}

impl Runner {
    /// Loads `path` and configures it (`kernel`: 0 df64, 1 f64, 2 mixed, 3 mixed64).
    pub fn load(path: &std::path::Path, kernel: u32) -> Result<Runner> {
        // SAFETY: a recurrence runner; its initializers only register an exit handler
        let library = unsafe { libloading::Library::new(path) }.map_err(|e| anyhow!("{}: {}", path.display(), e))?;
        let get = |name: &[u8]| -> Result<*const ()> {
            unsafe { library.get::<*const ()>(name) }
                .map(|s| *s)
                .map_err(|e| anyhow!("{}: {}", path.display(), e))
        };
        let run = get(b"pacman_recurrence_v1_run\0")?;
        let configure = get(b"pacman_recurrence_v1_configure\0")?;
        let symjit_kernel = get(b"pacman_recurrence_v1_symjit_kernel\0")?;
        let configure: unsafe extern "C" fn(*const Options) -> i32 = unsafe { std::mem::transmute(configure) };
        let options = Options { abi_version: 1, struct_size: size_of::<Options>() as u32, kernel, block: 0, no_sort: 0, stats: 0 };
        if unsafe { configure(&options) } != 0 {
            bail!("{}: configure failed", path.display());
        }
        Ok(Runner {
            run: unsafe { std::mem::transmute(run) },
            symjit_kernel: unsafe { std::mem::transmute(symjit_kernel) },
            _library: library,
        })
    }

    fn kernel_index(&self, sha256: &str) -> Result<u32> {
        let name = std::ffi::CString::new(sha256)?;
        let index = unsafe { (self.symjit_kernel)(name.as_ptr()) };
        u32::try_from(index).map_err(|_| anyhow!("the runner has no symjit kernel {sha256}"))
    }
}

/// `words` with the kernel index of each op 14 binding mapped by `map` (a recording's index to
/// the runner's). Op 14 is followed by its data words; every other op is one word.
pub fn remap_kernels(words: &[u64], map: &[u32]) -> Result<Vec<u64>> {
    let mut out = words.to_vec();
    let mut n = 0;
    while n < out.len() {
        let w = out[n];
        if w & 15 == 14 {
            let kernel = ((w >> 4) & 0xffff) as usize;
            let codes = ((w >> 20) & 0xff) + ((w >> 28) & 0xff);
            let to = *map.get(kernel).ok_or_else(|| anyhow!("op 14 names kernel {kernel}, the recording has {}", map.len()))?;
            out[n] = (w & !(0xffffu64 << 4)) | u64::from(to) << 4;
            n += 1 + codes.div_ceil(2) as usize;
        } else {
            n += 1;
        }
    }
    Ok(out)
}

/// The values of `r` replayed on `runner` (layout `[point][helicity]`).
pub fn replay(runner: &Runner, r: &Recorded) -> Result<Vec<f64>> {
    let rec = &r.recording;
    let map: Vec<u32> = rec.symjit_sha256.iter().map(|s| runner.kernel_index(s)).collect::<Result<_>>()?;
    let (n, h) = (r.external_count, r.helicity_count);
    let points_total = r.values.len() / h.max(1);
    let mut values = vec![0.0f64; r.values.len()];
    let mut tile_start = 0usize;
    for (k, run) in rec.runs.iter().enumerate() {
        if !run.upload_amplitude.is_empty() {
            bail!("run {k} uploads amplitudes (the recording has no host amplitudes)");
        }
        let words = remap_kernels(&run.words, &map)?;
        let (points, stride) = (run.points as usize, run.stride as usize);
        // the external momenta of the tile, source-major, crossed as RustiCol reads them
        let mut external = vec![0.0f64; n * 4 * points];
        for p in 0..points {
            for leg in 0..n {
                let (source, sign) = r.crossing.as_ref().map_or((leg, 1.0), |c| c[leg]);
                for c in 0..4 {
                    external[(leg * 4 + c) * points + p] = sign * r.momenta[((tile_start + p) * n + source) * 4 + c];
                }
            }
        }
        let mut current_re = vec![0.0f64; run.current_planes as usize * stride];
        let mut current_im = current_re.clone();
        let mut amplitude_re = vec![0.0f64; run.amplitude_planes as usize * stride];
        let mut amplitude_im = amplitude_re.clone();
        let momenta = vec![0.0f64; run.momentum_planes as usize * stride];
        let call = RunnerCall {
            abi_version: 1,
            struct_size: size_of::<RunnerCall>() as u32,
            words: words.as_ptr(),
            words_key: 0,
            word_count: words.len() as u32,
            width: run.width,
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            amplitude_re: amplitude_re.as_mut_ptr(),
            amplitude_im: amplitude_im.as_mut_ptr(),
            momenta: momenta.as_ptr(),
            scale_re: run.scale_re.as_ptr(),
            scale_im: run.scale_im.as_ptr(),
            factor_re: run.factor_re.as_ptr(),
            factor_im: run.factor_im.as_ptr(),
            upload_current: std::ptr::null(),
            upload_amplitude: std::ptr::null(),
            download_amplitude: run.download_amplitude.as_ptr(),
            download_current: std::ptr::null(),
            current_planes: run.current_planes,
            amplitude_planes: run.amplitude_planes,
            momentum_planes: run.momentum_planes,
            scale_count: run.scale_re.len() as u32,
            factor_count: run.factor_re.len() as u32,
            upload_current_runs: 0,
            upload_amplitude_runs: 0,
            download_amplitude_runs: (run.download_amplitude.len() / 2) as u32,
            download_current_runs: 0,
            stride: run.stride,
            points: run.points,
            zero_amplitude: run.zero_amplitude.as_ptr(),
            zero_amplitude_runs: (run.zero_amplitude.len() / 2) as u32,
            flags: FORMS_ON_DEVICE,
            external_momenta: external.as_ptr(),
            external_count: n as u32,
            term_count: run.term_slots.len() as u32,
            form_terms: run.form_terms.as_ptr(),
            term_slots: run.term_slots.as_ptr(),
            term_coefficients: run.term_coefficients.as_ptr(),
        };
        // SAFETY: every pointer addresses a live buffer of the stated size during the call
        let status = unsafe { (runner.run)(&call) };
        if status != 0 {
            bail!("run {k}: the runner returned {status}");
        }
        for red in &run.reductions {
            let plane = red.plane as usize * stride;
            for p in 0..points {
                let (re, im) = (amplitude_re[plane + p], amplitude_im[plane + p]);
                values[(tile_start + p) * h + red.helicity as usize] += red.weight * re.mul_add(re, im * im);
            }
        }
        tile_start += points;
    }
    if tile_start != points_total {
        bail!("the runs cover {tile_start} points of {points_total}");
    }
    let (num, den) = r.group_factor;
    if num != den {
        for v in values.iter_mut() {
            *v = *v * num as f64 / den as f64;
        }
    }
    Ok(values)
}
