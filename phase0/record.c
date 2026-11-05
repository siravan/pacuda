/* SPDX-License-Identifier: 0BSD */
/* Phase 0: a recording recurrence runner. It exports the runner's three entry points, logs
 * every call as one JSON line to $PACUDA_TRACE (default pacuda-trace.jsonl), and forwards it to
 * the real runner, $PACUDA_REAL_RUNNER. PACUDA_TRACE=none: no log (a pass-through that only
 * configures the runner from PACMAN_RECURRENCE_KERNEL, for the RustiCol hook path).
 *
 * Each line separates what a replay file would have to hold ("static": the words, widths, run
 * lists, table sizes and hashes of the scale, factor and form tables) from what changes with
 * the points ("data": point count, hashes of the momenta in, the amplitudes out, the uploaded
 * currents). Hashes are FNV-1a 64 of the bytes. */
#define _GNU_SOURCE
#include "../runner/pacman_recurrence.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *out;
static unsigned long seq;
static int (*real_run)(const pacman_recurrence_call *);
static int (*real_reduce)(const pacman_recurrence_reduce *);
static uint32_t (*real_capabilities)(void);
static int (*real_symjit_kernel)(const char *);

static void init(void) {
    if (real_run) return;
    const char *path = getenv("PACUDA_REAL_RUNNER");
    void *lib = path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : NULL;
    if (!lib) {
        fprintf(stderr, "pacuda record: PACUDA_REAL_RUNNER: %s\n", path ? dlerror() : "unset");
        abort();
    }
    real_run = dlsym(lib, "pacman_recurrence_v1_run");
    real_reduce = dlsym(lib, "pacman_recurrence_v1_reduce");
    real_capabilities = dlsym(lib, "pacman_recurrence_v1_capabilities");
    real_symjit_kernel = dlsym(lib, "pacman_recurrence_v1_symjit_kernel");  /* optional */
    /* development tooling: the kernel from PACMAN_RECURRENCE_KERNEL, as the runner's
     * configuration (the runner itself reads no environment variables) */
    int (*configure)(const pacman_recurrence_options *) = dlsym(lib, "pacman_recurrence_v1_configure");
    const char *k = getenv("PACMAN_RECURRENCE_KERNEL");
    if (configure && k && *k) {
        const char *st = getenv("PACMAN_RUNNER_STATS");
        pacman_recurrence_options o = {PACMAN_RECURRENCE_ABI, sizeof o, PACMAN_RECURRENCE_KERNEL_DF64, 0, 0,
                                       st && !strcmp(st, "1")};
        if (!strcmp(k, "f64")) o.kernel = PACMAN_RECURRENCE_KERNEL_F64;
        else if (!strcmp(k, "mixed")) o.kernel = PACMAN_RECURRENCE_KERNEL_MIXED;
        else if (!strcmp(k, "mixed64")) o.kernel = PACMAN_RECURRENCE_KERNEL_MIXED64;
        if (configure(&o) != 0) abort();
    }
    const char *trace = getenv("PACUDA_TRACE");
    out = trace && !strcmp(trace, "none") ? NULL : fopen(trace ? trace : "pacuda-trace.jsonl", "a");
    if ((!out && !(trace && !strcmp(trace, "none"))) || !real_run || !real_reduce || !real_capabilities) abort();
}

static uint64_t fnv(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = data;
    if (!p) return h;
    for (size_t i = 0; i < bytes; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
#define FNV0 1469598103934665603ull

static void runs_json(const char *name, const uint32_t *runs, uint32_t count) {
    fprintf(out, ",\"%s\":[", name);
    for (uint32_t i = 0; i < 2 * count; i++) fprintf(out, i ? ",%u" : "%u", runs[i]);
    fputc(']', out);
}

/* hash of the planes named by runs, `points` values each, at `stride` */
static uint64_t planes_hash(const double *re, const double *im, const uint32_t *runs,
                            uint32_t count, uint32_t stride, uint32_t points) {
    uint64_t h = FNV0;
    if (!re) return 0;
    for (uint32_t r = 0; r < count; r++)
        for (uint32_t p = runs[2 * r]; p < runs[2 * r] + runs[2 * r + 1]; p++) {
            h = fnv(h, re + (size_t)p * stride, points * sizeof(double));
            if (im) h = fnv(h, im + (size_t)p * stride, points * sizeof(double));
        }
    return h;
}

static long tid(void) { return syscall(SYS_gettid); }

uint32_t pacman_recurrence_v1_capabilities(void) {
    pthread_mutex_lock(&lock);
    init();
    pthread_mutex_unlock(&lock);
    return real_capabilities();
}

int pacman_recurrence_v1_symjit_kernel(const char *sha256_hex) {
    pthread_mutex_lock(&lock);
    init();
    pthread_mutex_unlock(&lock);
    return real_symjit_kernel ? real_symjit_kernel(sha256_hex) : -1;
}

int pacman_recurrence_v1_run(const pacman_recurrence_call *c) {
    pthread_mutex_lock(&lock);
    init();
    if (!out) {
        pthread_mutex_unlock(&lock);
        return real_run(c);
    }
    /* the struct may end before the reduction fields or before the forms (struct_size) */
    size_t size = c->struct_size;
    int has_zero = size > offsetof(pacman_recurrence_call, flags);
    int has_forms = size > offsetof(pacman_recurrence_call, term_coefficients);
    uint32_t flags = has_zero ? c->flags : 0;
    int forms = has_forms && (flags & PACMAN_RECURRENCE_FORMS_ON_DEVICE);
    uint64_t h_up = planes_hash(c->current_re, c->current_im, c->upload_current,
                                c->upload_current_runs, c->stride, c->points);
    uint64_t h_mom = forms ? 0 : fnv(FNV0, c->momenta, (size_t)c->momentum_planes * c->stride * 8);
    uint64_t h_ext = forms ? fnv(FNV0, c->external_momenta,
                                 (size_t)c->external_count * 4 * c->points * 8) : 0;
    int status = real_run(c);
    uint64_t h_amp = planes_hash(c->amplitude_re, c->amplitude_im, c->download_amplitude,
                                 c->download_amplitude_runs, c->stride, c->points);
    if (flags & PACMAN_RECURRENCE_KEEP_AMPLITUDES) h_amp = 0; /* not copied back */
    uint64_t h_cur = planes_hash(c->current_re, c->current_im, c->download_current,
                                 c->download_current_runs, c->stride, c->points);
    fprintf(out, "{\"seq\":%lu,\"tid\":%ld,\"call\":\"run\",\"status\":%d,\"static\":{", seq++,
            tid(), status);
    fprintf(out, "\"key\":\"%016llx\",\"words\":%u,\"words_hash\":\"%016llx\",\"width\":[%u,%u,%u,%u,%u,%u]",
            (unsigned long long)c->words_key, c->word_count,
            (unsigned long long)fnv(FNV0, c->words, (size_t)c->word_count * 8), c->width[0],
            c->width[1], c->width[2], c->width[3], c->width[4], c->width[5]);
    fprintf(out, ",\"planes\":[%u,%u,%u],\"scales\":%u,\"factors\":%u", c->current_planes,
            c->amplitude_planes, c->momentum_planes, c->scale_count, c->factor_count);
    uint64_t hs = fnv(fnv(FNV0, c->scale_re, c->scale_count * 8ul), c->scale_im, c->scale_count * 8ul);
    uint64_t hf = fnv(fnv(FNV0, c->factor_re, c->factor_count * 8ul), c->factor_im, c->factor_count * 8ul);
    fprintf(out, ",\"scales_hash\":\"%016llx\",\"factors_hash\":\"%016llx\",\"flags\":%u",
            (unsigned long long)hs, (unsigned long long)hf, flags);
    runs_json("upload_current", c->upload_current, c->upload_current_runs);
    runs_json("upload_amplitude", c->upload_amplitude, c->upload_amplitude_runs);
    runs_json("download_amplitude", c->download_amplitude, c->download_amplitude_runs);
    runs_json("download_current", c->download_current, c->download_current_runs);
    if (has_zero) runs_json("zero_amplitude", c->zero_amplitude, c->zero_amplitude_runs);
    if (forms) {
        uint32_t nforms = c->momentum_planes / 4;
        uint64_t h = fnv(FNV0, c->form_terms, nforms * 2 * 4ul);
        h = fnv(h, c->term_slots, c->term_count * 4ul);
        h = fnv(h, c->term_coefficients, c->term_count * 8ul);
        fprintf(out, ",\"externals\":%u,\"terms\":%u,\"forms_hash\":\"%016llx\"",
                c->external_count, c->term_count, (unsigned long long)h);
    }
    fprintf(out, "},\"data\":{\"points\":%u,\"stride\":%u,\"momenta_hash\":\"%016llx\","
            "\"external_hash\":\"%016llx\",\"uploaded_currents_hash\":\"%016llx\","
            "\"amplitudes_hash\":\"%016llx\",\"currents_hash\":\"%016llx\"}}\n",
            c->points, c->stride, (unsigned long long)h_mom, (unsigned long long)h_ext,
            (unsigned long long)h_up, (unsigned long long)h_amp, (unsigned long long)h_cur);
    fflush(out);
    pthread_mutex_unlock(&lock);
    return status;
}

int pacman_recurrence_v1_reduce(const pacman_recurrence_reduce *r) {
    pthread_mutex_lock(&lock);
    init();
    if (!out) {
        pthread_mutex_unlock(&lock);
        return real_reduce(r);
    }
    int zero_before = 1;
    for (uint32_t i = 0; i < r->points; i++) zero_before &= r->output[i] == 0.0;
    int status = real_reduce(r);
    uint64_t hp = fnv(FNV0, r->planes, r->count * 4ul);
    uint64_t hw = fnv(FNV0, r->weights, r->count * 8ul);
    fprintf(out, "{\"seq\":%lu,\"tid\":%ld,\"call\":\"reduce\",\"status\":%d,\"static\":{"
            "\"count\":%u,\"planes_hash\":\"%016llx\",\"weights_hash\":\"%016llx\"},"
            "\"data\":{\"points\":%u,\"output_zero_before\":%d,\"output_hash\":\"%016llx\","
            "\"output0\":%.17g}}\n",
            seq++, tid(), status, r->count, (unsigned long long)hp, (unsigned long long)hw,
            r->points, zero_before, (unsigned long long)fnv(FNV0, r->output, r->points * 8ul),
            r->points ? r->output[0] : 0.0);
    fflush(out);
    pthread_mutex_unlock(&lock);
    return status;
}
