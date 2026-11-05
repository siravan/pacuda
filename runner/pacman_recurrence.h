/* SPDX-License-Identifier: 0BSD */
/* The recurrence runner's C interface (librecurrence.so, recurrence_runner.cu). */
#ifndef PACMAN_RECURRENCE_H
#define PACMAN_RECURRENCE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define PACMAN_RECURRENCE_ABI 1

enum { W_P, W_A, W_F, W_X, W_S, W_K, W_COUNT };

// Planes are plane-major with `stride` doubles per plane, as RustiCol's arena. A run list holds
// pairs (first plane, plane count). The current arena stays on the device between calls: only
// the runs named are copied.
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint64_t *words;
    uint64_t words_key;            // nonzero: `words` never changes under this key (cached)
    uint32_t word_count;
    uint8_t width[8];              // W_P..W_K
    double *current_re, *current_im;
    double *amplitude_re, *amplitude_im;
    const double *momenta;         // momentum_planes planes (form-major, 4 per form)
    const double *scale_re, *scale_im;
    const double *factor_re, *factor_im;
    const uint32_t *upload_current, *upload_amplitude, *download_amplitude, *download_current;
    uint32_t current_planes, amplitude_planes, momentum_planes, scale_count, factor_count;
    uint32_t upload_current_runs, upload_amplitude_runs, download_amplitude_runs,
        download_current_runs;
    uint32_t stride, points;
    // since the reduction: a caller may pass a struct that ends here (the fields below are 0)
    const uint32_t *zero_amplitude;  // amplitude runs set to zero on the device, not uploaded
    uint32_t zero_amplitude_runs;
    uint32_t flags;                  // PACMAN_RECURRENCE_KEEP_AMPLITUDES, _FORMS_ON_DEVICE
    // since the forms on the device: a caller may pass a struct that ends above
    // PACMAN_RECURRENCE_FORMS_ON_DEVICE: `momenta` is not read; the momentum forms are filled on the
    // device from the external momenta, as RustiCol's `fill_momentum_form_from_external`: form f,
    // component k = 0.0 + c0 * p(s0, k) + c1 * p(s1, k) + ..., terms form_terms[2f] ..
    // + form_terms[2f+1] of (term_slots, term_coefficients); p(s, k) at
    // external_momenta[(s * 4 + k) * points + i] (source-major planes of `points` values)
    const double *external_momenta;
    uint32_t external_count;
    uint32_t term_count;
    const uint32_t *form_terms;      // (first term, term count) per form, momentum_planes / 4 forms
    const uint32_t *term_slots;
    const double *term_coefficients;
} pacman_recurrence_call;

// the download_amplitude runs stay on the device (not copied back) for pacman_recurrence_v1_reduce
#define PACMAN_RECURRENCE_KEEP_AMPLITUDES 1u
// the momentum forms are filled on the device from `external_momenta` (see above)
#define PACMAN_RECURRENCE_FORMS_ON_DEVICE 2u

// After a run with PACMAN_RECURRENCE_KEEP_AMPLITUDES: for each point i < points,
//   output[i] += weights[k] * fma(re, re, im * im)   for k = 0 .. count-1 in order,
// with (re, im) the run's amplitude plane planes[k] at point i: RustiCol's LC diagonal reduction
// (`accumulate_lc_diagonal_amplitude`), operation for operation. `output` is the caller's tile.
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *planes;
    const double *weights;
    double *output;
    uint32_t count, points;
} pacman_recurrence_reduce;

enum {
    PACMAN_RECURRENCE_OK = 0,
    PACMAN_RECURRENCE_INVALID = 2,
    PACMAN_RECURRENCE_DEVICE_ERROR = 3,
};

// After a run with PACMAN_RECURRENCE_KEEP_AMPLITUDES: several outputs in one call. For output
// o < outputs and point i < points,
//   output[o * output_stride + i * point_stride] += weights[k] * fma(re, re, im * im)
// for k = offsets[o] .. offsets[o+1]-1 in order, (re, im) the run's amplitude plane planes[k] at
// point i: RustiCol's resolved LC reduction (one output per helicity; `[point][helicity]` is
// output_stride 1, point_stride the helicity count). One call reduces a kept run (it ends it).
// flags PACMAN_RECURRENCE_OUTPUT_ZERO: the output starts at zero (cleared on the device, not
// uploaded; only the results come back).
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *planes;
    const double *weights;
    const uint32_t *offsets;  // outputs + 1 entries
    double *output;
    uint32_t outputs, points, output_stride, point_stride;
    uint32_t flags;
} pacman_recurrence_reduce_resolved;
#define PACMAN_RECURRENCE_OUTPUT_ZERO 1u

// After a run with PACMAN_RECURRENCE_KEEP_AMPLITUDES: the helicity sampling of the UMAMI LC
// pass, per point i < points. With v[k] the value of output k (as `reduce_resolved` from zero,
// times numerator / denominator when they differ, times scale[i] when scale is given and it is
// not 1), in the order of the outputs (their helicities ascending):
//   total[i] = 0 + v[0] + v[1] + ...,
//   target = random[i] * total[i]; chosen[i] = the helicity of the output at which the cumulative
//   sum of the positive v (in order) first exceeds target; if it never does, of the last output
//   with a positive v; with none, 0. As pyAmpliCol's umami.c samples. One call ends a kept run.
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *planes;
    const double *weights;
    const uint32_t *offsets;   // outputs + 1 entries
    const uint32_t *helicity;  // per output, its helicity (what `chosen` reports)
    uint32_t outputs, points;
    const double *random;      // per point, in [0, 1)
    const double *scale;       // per point, or NULL (1)
    double numerator, denominator;
    double *total;             // per point
    uint32_t *chosen;          // per point
} pacman_recurrence_reduce_sample;

int pacman_recurrence_v1_run(const pacman_recurrence_call *call);
int pacman_recurrence_v1_reduce_resolved(const pacman_recurrence_reduce_resolved *reduce);
int pacman_recurrence_v1_reduce_sample(const pacman_recurrence_reduce_sample *sample);
// Pinned (page-locked) host memory for the buffers a caller passes (faster copies); NULL if none.
void *pacman_recurrence_v1_host_alloc(size_t bytes);
void pacman_recurrence_v1_host_free(void *p);
int pacman_recurrence_v1_reduce(const pacman_recurrence_reduce *reduce);
// bit 0: op 12 (sources on the device); bit 1: PACMAN_RECURRENCE_FORMS_ON_DEVICE
#define PACMAN_RECURRENCE_CAP_SOURCES 1u
#define PACMAN_RECURRENCE_CAP_FORMS 2u
// The runner's configuration: before the first run; afterwards only the same one again:
typedef struct {
    uint32_t abi_version;  // PACMAN_RECURRENCE_ABI
    uint32_t struct_size;
    uint32_t kernel;       // PACMAN_RECURRENCE_KERNEL_*
    uint32_t block;        // threads per block, 32..1024 (0: 128)
    uint32_t no_sort;      // 1: keep the contribution rows in plan order
    uint32_t stats;        // 1: print the statistics at exit
} pacman_recurrence_options;
enum {
    PACMAN_RECURRENCE_KERNEL_DF64 = 0,  // the default
    PACMAN_RECURRENCE_KERNEL_F64 = 1,
    PACMAN_RECURRENCE_KERNEL_MIXED = 2,
    PACMAN_RECURRENCE_KERNEL_MIXED64 = 3,
};
int pacman_recurrence_v1_configure(const pacman_recurrence_options *options);
// bit 2: op 13/14 (symjit rows), with generated kernels (`pacman_recurrence_v1_symjit_kernel`)
#define PACMAN_RECURRENCE_CAP_SYMJIT 4u
uint32_t pacman_recurrence_v1_capabilities(void);
// the index of the generated symjit kernel of this SHA-256 (hex), or -1
int pacman_recurrence_v1_symjit_kernel(const char *sha256_hex);

#ifdef __cplusplus
}
#endif
#endif
