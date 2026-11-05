// SPDX-License-Identifier: 0BSD
//! symjit227 CHECK.bin FILE.plane.symjit... : for each kernel of a `pacuda-gen check` file (in
//! its order; the files deduplicated as `check` does), symjit 2.27.3's scalar and SIMD outputs on
//! its inputs, compared bit for bit with the 2.28.3 outputs stored there. With OUT.bin after
//! CHECK.bin (a `.bin` name), also writes a copy of CHECK.bin whose SIMD outputs are 2.27.3's.
use anyhow::{anyhow, Result};
use symjit::{Application, Compiled, CompilerType, Config, Defuns, PlaneDescriptor};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bin = std::fs::read(&args[0])?;
    let (out_path, first) = match args.get(1) {
        Some(a) if a.ends_with(".bin") => (Some(a.clone()), 2),
        _ => (None, 1),
    };
    let mut copy = bin.clone();
    let mut files: Vec<Vec<u8>> = Vec::new();
    for f in &args[first..] {
        let b = std::fs::read(f)?;
        if !files.contains(&b) {
            files.push(b);
        }
    }
    let u32_at = |o: usize| u32::from_le_bytes(bin[o..o + 4].try_into().unwrap()) as usize;
    let f64s = |o: usize, n: usize| -> Vec<f64> {
        (0..n).map(|k| f64::from_le_bytes(bin[o + 8 * k..o + 8 * k + 8].try_into().unwrap())).collect()
    };
    let (count, n) = (u32_at(0), u32_at(4));
    let mut o = 8;
    let mut total = (0usize, 0usize);
    for k in 0..count {
        let (ni, no) = (u32_at(o), u32_at(o + 4));
        o += 8;
        let inputs = f64s(o, ni * n);
        o += 8 * ni * n;
        let scalar28 = f64s(o, no * n);
        o += 8 * no * n;
        let simd28 = f64s(o, no * n);
        let simd_at = o;
        o += 8 * no * n;
        let mut config = Config::new(CompilerType::Native, 0)?;
        config.set_opt_level(0);
        config.set_cse(false);
        config.set_fastmath(false);
        config.set_simd(false);
        config.enable_simd512(false);
        config.set_simd_branch(false);
        config.set_complex(false);
        config.set_fast_complex(false);
        config.set_threads(false);
        config.set_symbolica(false);
        config.set_compact(false);
        config.set_compress(false);
        config.set_direct(false);
        config.set_huge(false);
        config.set_parallel_mul(false);
        config.set_direct_arena(false);
        config.set_defuns(Defuns::new());
        let mut input = &files[k][..];
        let mut app = Application::load(&mut input, &config)?;
        app.prepare_simd();
        let applet = app.seal()?;
        let scalar = applet.scalar_plane_kernel().ok_or_else(|| anyhow!("no scalar kernel"))?;
        let simd = match applet.compiled_simd.as_ref() {
            Some(c) if c.support_indirect() => applet.simd_plane_kernel().map(|s| (s, c.count_lanes())),
            _ => None,
        };
        let run = |use_simd: bool| -> Vec<f64> {
            let mut inp = inputs.clone();
            let mut out = vec![0.0f64; no * n];
            let mut table: Vec<PlaneDescriptor<f64>> = Vec::new();
            for p in 0..ni {
                table.push(unsafe { PlaneDescriptor::from_raw_parts(inp.as_mut_ptr().add(p * n), n) });
            }
            for p in 0..no {
                table.push(unsafe { PlaneDescriptor::from_raw_parts(out.as_mut_ptr().add(p * n), n) });
            }
            let mut point = 0;
            if let (true, Some((kernel, lanes))) = (use_simd, simd) {
                while n - point >= lanes {
                    unsafe { kernel(std::ptr::null(), table.as_ptr(), point, applet.params.as_ptr()) };
                    point += lanes;
                }
            }
            while point < n {
                unsafe { scalar(std::ptr::null(), table.as_ptr(), point, applet.params.as_ptr()) };
                point += 1;
            }
            out
        };
        let (s27, v27) = (run(false), run(true));
        for (j, v) in v27.iter().enumerate() {
            copy[simd_at + 8 * j..simd_at + 8 * j + 8].copy_from_slice(&v.to_le_bytes());
        }
        let diff = |a: &[f64], b: &[f64]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let (ds, dv) = (diff(&s27, &scalar28), diff(&v27, &simd28));
        total.0 += ds;
        total.1 += dv;
        if ds + dv > 0 {
            println!("kernel {}: 2.27.3 vs 2.28.3: scalar differs in {}, SIMD in {} of {}", k, ds, dv, no * n);
        }
    }
    println!("{} kernels: 2.27.3 vs 2.28.3 differ in {} scalar and {} SIMD values", count, total.0, total.1);
    if let Some(p) = out_path {
        std::fs::write(p, copy)?;
    }
    Ok(())
}
