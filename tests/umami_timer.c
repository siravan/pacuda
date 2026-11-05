/* SPDX-License-Identifier: 0BSD */
/* A timing shim for a UMAMI library: it exports the nine UMAMI functions, forwards them to the
 * real library (REAL_LIB, loaded at start), and adds up the wall time of umami_matrix_element
 * (calls, events, seconds) and of umami_initialize. At exit it appends one line to LOG_FILE:
 *   NAME pid calls events matrix_element_seconds initialize_seconds
 * Built by tests/time_me.py with -DREAL_LIB, -DLOG_FILE and -DNAME. */
#define _GNU_SOURCE
#include "umami.h"
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void *real;
static atomic_ullong me_ns, me_calls, me_events, init_ns;

#define FN(name) static __typeof__(&name) real_##name;
FN(umami_get_meta) FN(umami_supported_inputs) FN(umami_required_inputs) FN(umami_supported_outputs)
FN(umami_initialize) FN(umami_get_parameter) FN(umami_set_parameter) FN(umami_matrix_element)
FN(umami_free)

static unsigned long long now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000ull + (unsigned long long)t.tv_nsec;
}

__attribute__((constructor)) static void load(void) {
    real = dlopen(REAL_LIB, RTLD_NOW | RTLD_LOCAL);
    if (!real) { fprintf(stderr, "umami_timer: %s\n", dlerror()); abort(); }
#define GET(name) real_##name = (__typeof__(real_##name))dlsym(real, #name); \
    if (!real_##name) { fprintf(stderr, "umami_timer: %s: no %s\n", REAL_LIB, #name); abort(); }
    GET(umami_get_meta) GET(umami_supported_inputs) GET(umami_required_inputs) GET(umami_supported_outputs)
    GET(umami_initialize) GET(umami_get_parameter) GET(umami_set_parameter) GET(umami_matrix_element)
    GET(umami_free)
}

__attribute__((destructor)) static void report(void) {
    if (atomic_load(&me_calls) == 0 && atomic_load(&init_ns) == 0) return;
    FILE *f = fopen(LOG_FILE, "a");
    if (!f) return;
    fprintf(f, "%s %d %llu %llu %.6f %.6f\n", NAME, (int)getpid(), atomic_load(&me_calls),
            atomic_load(&me_events), atomic_load(&me_ns) * 1e-9, atomic_load(&init_ns) * 1e-9);
    fclose(f);
}

UmamiStatus umami_get_meta(UmamiMetaKey key, void *result) { return real_umami_get_meta(key, result); }
UmamiStatus umami_supported_inputs(bool const **s, int *c) { return real_umami_supported_inputs(s, c); }
UmamiStatus umami_required_inputs(bool const **s, int *c) { return real_umami_required_inputs(s, c); }
UmamiStatus umami_supported_outputs(bool const **s, int *c) { return real_umami_supported_outputs(s, c); }
UmamiStatus umami_initialize(UmamiHandle *handle, char const *path) {
    unsigned long long t = now();
    UmamiStatus status = real_umami_initialize(handle, path);
    atomic_fetch_add(&init_ns, now() - t);
    return status;
}
UmamiStatus umami_get_parameter(UmamiHandle h, char const *n, double *re, double *im) {
    return real_umami_get_parameter(h, n, re, im);
}
UmamiStatus umami_set_parameter(UmamiHandle h, char const *n, double re, double im) {
    return real_umami_set_parameter(h, n, re, im);
}
UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const *input_keys, void const *const *inputs,
    size_t output_count, UmamiOutputKey const *output_keys, void *const *outputs) {
    unsigned long long t = now();
    UmamiStatus status = real_umami_matrix_element(handle, count, stride, offset, input_count, input_keys,
                                                   inputs, output_count, output_keys, outputs);
    atomic_fetch_add(&me_ns, now() - t);
    atomic_fetch_add(&me_calls, 1);
    atomic_fetch_add(&me_events, count);
    return status;
}
UmamiStatus umami_free(UmamiHandle handle) { return real_umami_free(handle); }
