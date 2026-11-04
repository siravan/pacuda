/* SPDX-License-Identifier: 0BSD */
/* The recurrence runner's C interface (librecurrence.so, recurrence_runner.cu). */
#ifndef PACMAN_RECURRENCE_H
#define PACMAN_RECURRENCE_H
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

int pacman_recurrence_v1_run(const pacman_recurrence_call *call);
int pacman_recurrence_v1_reduce(const pacman_recurrence_reduce *reduce);
// bit 0: op 12 (sources on the device); bit 1: PACMAN_RECURRENCE_FORMS_ON_DEVICE
#define PACMAN_RECURRENCE_CAP_SOURCES 1u
#define PACMAN_RECURRENCE_CAP_FORMS 2u
uint32_t pacman_recurrence_v1_capabilities(void);

#ifdef __cplusplus
}
#endif
#endif
