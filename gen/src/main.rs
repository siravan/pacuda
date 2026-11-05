// SPDX-License-Identifier: 0BSD
//! pacuda-gen: CUDA device functions from pyAmpliCol's symjit plane kernels.
//!
//!   pacuda-gen cuda FILE.plane.symjit... [-o OUT.cuh]
//!   pacuda-gen check FILE.plane.symjit... -d DIR [-n POINTS]
//!
//! `cuda` writes one function per kernel, named `pk_<NNNNNN>` after the kernel directory
//! (`kernels/NNNNNN/application-1.plane.symjit`). `check` also writes, in DIR, `kernels.cuh`,
//! `check.bin` (random inputs and symjit's CPU outputs: scalar kernel, and SIMD kernel in lane
//! blocks with the scalar one for the rest, as RustiCol calls it) and `check.cu` (runs the
//! functions on the GPU and compares them with both, bit for bit). Build it with
//! `nvcc -O3 -fmad=false -gencode=arch=compute_XX,code=sm_XX -o check check.cu` (the GPU's SM) and
//! run `./check check.bin`.

mod cuda;
mod sjb;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use symjit::{Application, CompilerType, Config, Defuns, PlaneDescriptor};

fn kernel_name(path: &Path) -> String {
    let dir = path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).unwrap_or("");
    if !dir.is_empty() && dir.chars().all(|c| c.is_ascii_digit()) {
        format!("pk_{}", dir)
    } else {
        let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("kernel");
        let clean: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        format!("pk_{}", clean)
    }
}

fn generate(paths: &[PathBuf]) -> Result<(String, Vec<cuda::CudaFunction>)> {
    let mut source = String::from(cuda::PRELUDE);
    let mut functions = Vec::new();
    for path in paths {
        let bytes = std::fs::read(path).map_err(|e| anyhow!("{}: {}", path.display(), e))?;
        let mut model = sjb::SavedModel::read(&bytes).map_err(|e| anyhow!("{}: {:#}", path.display(), e))?;
        let f = model.cuda(&kernel_name(path))?;
        source.push('\n');
        source.push_str(&f.source);
        functions.push(f);
    }
    Ok((source, functions))
}

/// symjit's CPU kernels of a plane application, loaded as RustiCol loads them (the saved
/// configuration replaces this all-off one; from symjit-utils `replay.rs`).
struct CpuKernel {
    _applet: symjit::Applet,
    params: Vec<f64>,
    scalar: symjit::CompiledPlaneFunc<f64>,
    simd: Option<(symjit::CompiledPlaneFunc<f64>, usize)>,
}

fn cpu_kernel(bytes: &[u8]) -> Result<CpuKernel> {
    use symjit::Compiled;
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
    let mut input = bytes;
    let mut app = Application::load(&mut input, &config)?;
    app.prepare_simd();
    let applet = app.seal()?;
    let scalar = applet.scalar_plane_kernel().ok_or_else(|| anyhow!("no scalar plane kernel"))?;
    let simd = match applet.compiled_simd.as_ref() {
        Some(c) if c.support_indirect() => applet.simd_plane_kernel().map(|k| (k, c.count_lanes())),
        _ => None,
    };
    Ok(CpuKernel { params: applet.params.clone(), _applet: applet, scalar, simd })
}

fn check(paths: &[PathBuf], dir: &Path, points: usize) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let (source, functions) = generate(paths)?;
    std::fs::write(dir.join("kernels.cuh"), &source)?;
    let mut bin: Vec<u8> = Vec::new();
    let put = |bin: &mut Vec<u8>, v: &[f64]| v.iter().for_each(|x| bin.extend_from_slice(&x.to_le_bytes()));
    bin.extend_from_slice(&(functions.len() as u32).to_le_bytes());
    bin.extend_from_slice(&(points as u32).to_le_bytes());
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        // values in (-2, 2), of all magnitudes down to 1e-3
        let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
        let m = 10f64.powf(-3.0 * ((seed & 0xff) as f64 / 255.0));
        (4.0 * u - 2.0) * m.max(1e-3)
    };
    for (path, f) in paths.iter().zip(&functions) {
        let k = cpu_kernel(&std::fs::read(path)?)?;
        let (n_in, n_out) = (f.shape.count_in, f.shape.count_out);
        let inputs: Vec<f64> = (0..n_in * points).map(|_| random()).collect();
        let run = |simd: bool| -> Result<Vec<f64>> {
            let mut inp = inputs.clone();
            let mut out = vec![0.0f64; n_out * points];
            let mut table: Vec<PlaneDescriptor<f64>> = Vec::new();
            for p in 0..n_in {
                table.push(unsafe { PlaneDescriptor::from_raw_parts(inp.as_mut_ptr().add(p * points), points) });
            }
            for p in 0..n_out {
                table.push(unsafe { PlaneDescriptor::from_raw_parts(out.as_mut_ptr().add(p * points), points) });
            }
            let mut point = 0;
            if let (true, Some((kernel, lanes))) = (simd, k.simd) {
                while points - point >= lanes {
                    let status = unsafe { kernel(std::ptr::null(), table.as_ptr(), point, k.params.as_ptr()) };
                    if status != 0 {
                        bail!("{}: SIMD status {}", f.name, status);
                    }
                    point += lanes;
                }
            }
            while point < points {
                let status = unsafe { (k.scalar)(std::ptr::null(), table.as_ptr(), point, k.params.as_ptr()) };
                if status != 0 {
                    bail!("{}: scalar status {}", f.name, status);
                }
                point += 1;
            }
            Ok(out)
        };
        let scalar = run(false)?;
        let simd = run(true)?;
        eprintln!(
            "{}: {} inputs, {} outputs, SIMD {}, SIMD == scalar: {}",
            f.name,
            n_in,
            n_out,
            k.simd.map_or("none".to_string(), |(_, l)| format!("{} lanes", l)),
            scalar.iter().zip(&simd).all(|(a, b)| a.to_bits() == b.to_bits())
        );
        bin.extend_from_slice(&(n_in as u32).to_le_bytes());
        bin.extend_from_slice(&(n_out as u32).to_le_bytes());
        put(&mut bin, &inputs);
        put(&mut bin, &scalar);
        put(&mut bin, &simd);
    }
    std::fs::write(dir.join("check.bin"), &bin)?;

    let mut cu = String::new();
    let _ = writeln!(cu, "// Generated by pacuda-gen check: the device functions against symjit's CPU kernels.");
    let _ = writeln!(cu, "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <cstdint>\n#include <vector>");
    let _ = writeln!(cu, "#include \"kernels.cuh\"");
    let _ = writeln!(cu, "__global__ void run(int k, int n_in, int n_out, int n, const double *in, double *out) {{");
    let _ = writeln!(cu, "\tint i = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (i >= n) return;");
    let _ = writeln!(cu, "\tdouble a[64], b[64];\n\tfor (int p = 0; p < n_in; p++) a[p] = in[p * n + i];");
    let _ = writeln!(cu, "\tswitch (k) {{");
    for (k, f) in functions.iter().enumerate() {
        let _ = writeln!(cu, "\tcase {}: {}(a, b); break;", k, f.name);
    }
    let _ = writeln!(cu, "\t}}\n\tfor (int p = 0; p < n_out; p++) out[p * n + i] = b[p];\n}}");
    let names: Vec<String> = functions.iter().map(|f| format!("\"{}\"", f.name)).collect();
    let _ = writeln!(cu, "static const char *names[] = {{{}}};", names.join(", "));
    cu.push_str(CHECK_MAIN);
    std::fs::write(dir.join("check.cu"), cu)?;
    if functions.iter().any(|f| f.shape.count_in > 64 || f.shape.count_out > 64) {
        bail!("check.cu takes at most 64 inputs and outputs per kernel");
    }
    eprintln!("wrote {}/{{kernels.cuh,check.bin,check.cu}}", dir.display());
    Ok(())
}

const CHECK_MAIN: &str = r#"
int main(int argc, char **argv) {
	FILE *f = fopen(argc > 1 ? argv[1] : "check.bin", "rb");
	if (!f) { perror("check.bin"); return 2; }
	uint32_t count, n;
	if (fread(&count, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) return 2;
	int bad = 0;
	for (uint32_t k = 0; k < count; k++) {
		uint32_t n_in, n_out;
		if (fread(&n_in, 4, 1, f) != 1 || fread(&n_out, 4, 1, f) != 1) return 2;
		std::vector<double> in((size_t)n_in * n), scalar((size_t)n_out * n), simd((size_t)n_out * n), gpu((size_t)n_out * n);
		if (fread(in.data(), 8, in.size(), f) != in.size() || fread(scalar.data(), 8, scalar.size(), f) != scalar.size() ||
		    fread(simd.data(), 8, simd.size(), f) != simd.size()) return 2;
		double *din, *dout;
		cudaMalloc(&din, in.size() * 8); cudaMalloc(&dout, gpu.size() * 8);
		cudaMemcpy(din, in.data(), in.size() * 8, cudaMemcpyHostToDevice);
		run<<<(n + 127) / 128, 128>>>(k, n_in, n_out, n, din, dout);
		cudaError_t e = cudaGetLastError();
		if (e == cudaSuccess) e = cudaDeviceSynchronize();
		if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", names[k], cudaGetErrorString(e)); return 77; }
		if (cudaMemcpy(gpu.data(), dout, gpu.size() * 8, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr, "CUDA error\n"); return 77; }
		cudaFree(din); cudaFree(dout);
		size_t vs_scalar = 0, vs_simd = 0;
		double worst = 0.0;
		for (size_t j = 0; j < gpu.size(); j++) {
			vs_scalar += memcmp(&gpu[j], &scalar[j], 8) != 0;
			vs_simd += memcmp(&gpu[j], &simd[j], 8) != 0;
			double scale = fabs(scalar[j]) > 1e-300 ? fabs(scalar[j]) : 1.0;
			double e = fabs(gpu[j] - scalar[j]) / scale;
			if (e > worst) worst = e;
		}
		printf("%s: %u inputs, %u outputs, %u points: differs from scalar in %zu, from SIMD in %zu of %zu values (max rel %.1e)\n",
		       names[k], n_in, n_out, n, vs_scalar, vs_simd, gpu.size(), worst);
		bad += vs_scalar != 0;
	}
	return bad ? 1 : 0;
}
"#;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = (|| -> Result<()> {
        let command = args.first().ok_or_else(|| anyhow!("usage: pacuda-gen cuda|check FILE... [-o OUT | -d DIR] [-n POINTS]"))?;
        let (mut files, mut out, mut dir, mut points) = (Vec::new(), None, None, 4096usize);
        let mut it = args[1..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "-o" => out = it.next().cloned(),
                "-d" => dir = it.next().cloned(),
                "-n" => points = it.next().ok_or_else(|| anyhow!("-n needs a value"))?.parse()?,
                _ => files.push(PathBuf::from(a)),
            }
        }
        match command.as_str() {
            "cuda" => {
                let (source, _) = generate(&files)?;
                match out {
                    Some(o) => std::fs::write(o, source)?,
                    None => print!("{}", source),
                }
                Ok(())
            }
            "check" => check(&files, Path::new(&dir.ok_or_else(|| anyhow!("check needs -d DIR"))?), points),
            other => bail!("unknown command `{}`", other),
        }
    })();
    if let Err(e) = result {
        eprintln!("pacuda-gen: {:#}", e);
        std::process::exit(1);
    }
}
