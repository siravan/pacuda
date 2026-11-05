// SPDX-License-Identifier: 0BSD
//! Evaluates a `.pacuda` run template on a recurrence runner (`librecurrence.so`), as the C
//! wrapper will: the points in tiles, each tile one runner call that keeps its amplitudes on the
//! device and one `reduce_resolved` into every helicity output (each output's planes in recorded
//! order; or, with `host_reduce`, the amplitudes copied back and reduced on the host as RustiCol
//! does), then the process-group factor (`apply_process_group_factor`).

use anyhow::{anyhow, bail, Result};

use crate::format::{Colour, Process};

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

/// `pacman_recurrence_reduce_resolved`.
#[repr(C)]
struct ReduceResolved {
    abi_version: u32,
    struct_size: u32,
    planes: *const u32,
    weights: *const f64,
    offsets: *const u32,
    output: *mut f64,
    outputs: u32,
    points: u32,
    output_stride: u32,
    point_stride: u32,
    flags: u32,
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

const KEEP_AMPLITUDES: u32 = 1;
const FORMS_ON_DEVICE: u32 = 2;

pub struct Runner {
    run: unsafe extern "C" fn(*const RunnerCall) -> i32,
    reduce_resolved: unsafe extern "C" fn(*const ReduceResolved) -> i32,
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
        let reduce_resolved = get(b"pacman_recurrence_v1_reduce_resolved\0")?;
        let configure = get(b"pacman_recurrence_v1_configure\0")?;
        let symjit_kernel = get(b"pacman_recurrence_v1_symjit_kernel\0")?;
        let configure: unsafe extern "C" fn(*const Options) -> i32 = unsafe { std::mem::transmute(configure) };
        let options = Options { abi_version: 1, struct_size: size_of::<Options>() as u32, kernel, block: 0, no_sort: 0, stats: 0 };
        if unsafe { configure(&options) } != 0 {
            bail!("{}: configure failed", path.display());
        }
        Ok(Runner {
            run: unsafe { std::mem::transmute(run) },
            reduce_resolved: unsafe { std::mem::transmute(reduce_resolved) },
            symjit_kernel: unsafe { std::mem::transmute(symjit_kernel) },
            _library: library,
        })
    }

    /// The runner's indices of the kernels `sha256` (a file's or a recording's table).
    pub fn kernel_map(&self, sha256: &[String]) -> Result<Vec<u32>> {
        sha256
            .iter()
            .map(|s| {
                let name = std::ffi::CString::new(s.as_str())?;
                let index = unsafe { (self.symjit_kernel)(name.as_ptr()) };
                u32::try_from(index).map_err(|_| anyhow!("the runner has no symjit kernel {s}"))
            })
            .collect()
    }
}

/// `words` with the kernel index of each op 14 binding mapped by `map`. Op 14 is followed by its
/// data words; every other op is one word.
pub fn remap_kernels(words: &[u64], map: &[u32]) -> Result<Vec<u64>> {
    let mut out = words.to_vec();
    let mut n = 0;
    while n < out.len() {
        let w = out[n];
        if w & 15 == 14 {
            let kernel = ((w >> 4) & 0xffff) as usize;
            let codes = ((w >> 20) & 0xff) + ((w >> 28) & 0xff);
            let to = *map.get(kernel).ok_or_else(|| anyhow!("op 14 names kernel {kernel}, the table has {}", map.len()))?;
            out[n] = (w & !(0xffffu64 << 4)) | u64::from(to) << 4;
            n += 1 + codes.div_ceil(2) as usize;
        } else {
            n += 1;
        }
    }
    Ok(out)
}

/// The values (`[point][helicity]`) of `colour` of `process` at `points` points of `momenta`
/// (point-major, the runtime's input order), in tiles of `tile` points; `map`: the runner's
/// indices of the file's symjit kernels.
pub fn evaluate(
    runner: &Runner,
    map: &[u32],
    process: &Process,
    colour: &Colour,
    momenta: &[f64],
    points: usize,
    tile: usize,
    host_reduce: bool,
) -> Result<Vec<f64>> {
    let t = &colour.template;
    let red = &colour.reduction;
    let (n, h) = (process.legs as usize, process.helicities as usize);
    if momenta.len() != points * n * 4 || tile == 0 {
        bail!("momenta of {} values for {points} points of {n} legs (tile {tile})", momenta.len());
    }
    let words = remap_kernels(&t.words, map)?;
    let mut values = vec![0.0f64; points * h];
    let stride = tile;
    let mut current_re = vec![0.0f64; t.current_planes as usize * stride];
    let mut current_im = current_re.clone();
    let mut amplitude_re = vec![0.0f64; t.amplitude_planes as usize * stride];
    let mut amplitude_im = amplitude_re.clone();
    let host_momenta = vec![0.0f64; t.momentum_planes as usize * stride];
    let mut external = vec![0.0f64; n * 4 * tile];
    let mut start = 0usize;
    while start < points {
        let count = tile.min(points - start);
        // the external momenta of the tile, source-major, crossed as RustiCol reads them
        for p in 0..count {
            for leg in 0..n {
                let (source, sign) = process.crossing.as_ref().map_or((leg, 1.0), |c| (c[leg].0 as usize, c[leg].1));
                for c in 0..4 {
                    external[(leg * 4 + c) * count + p] = sign * momenta[((start + p) * n + source) * 4 + c];
                }
            }
        }
        let call = RunnerCall {
            abi_version: 1,
            struct_size: size_of::<RunnerCall>() as u32,
            words: words.as_ptr(),
            words_key: 0,
            word_count: words.len() as u32,
            width: t.width,
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            amplitude_re: amplitude_re.as_mut_ptr(),
            amplitude_im: amplitude_im.as_mut_ptr(),
            momenta: host_momenta.as_ptr(),
            scale_re: t.scale_re.as_ptr(),
            scale_im: t.scale_im.as_ptr(),
            factor_re: t.factor_re.as_ptr(),
            factor_im: t.factor_im.as_ptr(),
            upload_current: std::ptr::null(),
            upload_amplitude: std::ptr::null(),
            download_amplitude: t.download_amplitude.as_ptr(),
            download_current: std::ptr::null(),
            current_planes: t.current_planes,
            amplitude_planes: t.amplitude_planes,
            momentum_planes: t.momentum_planes,
            scale_count: t.scale_re.len() as u32,
            factor_count: t.factor_re.len() as u32,
            upload_current_runs: 0,
            upload_amplitude_runs: 0,
            download_amplitude_runs: (t.download_amplitude.len() / 2) as u32,
            download_current_runs: 0,
            stride: stride as u32,
            points: count as u32,
            zero_amplitude: t.zero_amplitude.as_ptr(),
            zero_amplitude_runs: (t.zero_amplitude.len() / 2) as u32,
            flags: FORMS_ON_DEVICE | if host_reduce { 0 } else { KEEP_AMPLITUDES },
            external_momenta: external.as_ptr(),
            external_count: n as u32,
            term_count: t.term_slots.len() as u32,
            form_terms: t.form_terms.as_ptr(),
            term_slots: t.term_slots.as_ptr(),
            term_coefficients: t.term_coefficients.as_ptr(),
        };
        // SAFETY: every pointer addresses a live buffer of the stated size during the call
        let status = unsafe { (runner.run)(&call) };
        if status != 0 {
            bail!("tile at {start}: the runner returned {status}");
        }
        if host_reduce {
            for hel in 0..h {
                for k in red.offsets[hel] as usize..red.offsets[hel + 1] as usize {
                    let plane = red.planes[k] as usize * stride;
                    for p in 0..count {
                        let (re, im) = (amplitude_re[plane + p], amplitude_im[plane + p]);
                        values[(start + p) * h + hel] += red.weights[k] * re.mul_add(re, im * im);
                    }
                }
            }
        } else {
            let reduce = ReduceResolved {
                abi_version: 1,
                struct_size: size_of::<ReduceResolved>() as u32,
                planes: red.planes.as_ptr(),
                weights: red.weights.as_ptr(),
                offsets: red.offsets.as_ptr(),
                output: values[start * h..].as_mut_ptr(),
                outputs: h as u32,
                points: count as u32,
                output_stride: 1,
                point_stride: h as u32,
                flags: 0,
            };
            // SAFETY: the tables live during the call; the output covers the tile's points
            let status = unsafe { (runner.reduce_resolved)(&reduce) };
            if status != 0 {
                bail!("tile at {start}: the resolved reduction returned {status}");
            }
        }
        start += count;
    }
    let (num, den) = process.factor;
    if num != den {
        for v in values.iter_mut() {
            *v = *v * num as f64 / den as f64;
        }
    }
    Ok(values)
}

/// Compares `got` with `want` (`[point][helicity]`): (values differing in bits, the worst error
/// relative to the point's total, the worst relative error of a point's total).
pub fn compare(got: &[f64], want: &[f64], helicities: usize) -> (usize, f64, f64) {
    let differ = got.iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let h = helicities.max(1);
    let (mut worst, mut worst_total) = (0.0f64, 0.0f64);
    for (a, b) in got.chunks(h).zip(want.chunks(h)) {
        let scale: f64 = b.iter().map(|v| v.abs()).sum();
        if scale == 0.0 {
            continue;
        }
        for (x, y) in a.iter().zip(b) {
            worst = worst.max((x - y).abs() / scale);
        }
        let (ta, tb): (f64, f64) = (a.iter().sum(), b.iter().sum());
        worst_total = worst_total.max((ta / tb - 1.0).abs());
    }
    (differ, worst, worst_total)
}
