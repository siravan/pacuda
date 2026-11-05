/* SPDX-License-Identifier: 0BSD */
/* pacuda: the UMAMI interface of a pyAmpliCol LC artifact on the recurrence runner.
 *
 * A drop-in for pyAmpliCol's `umami.c` (same provider header, same nine functions, same inputs
 * and outputs), with no RustiCol: the recurrence plan runs of each (process, colour id) were
 * recorded into `<artifact>.pacuda` by `pacuda-convert`, and are run here on
 * `librecurrence.so` (linked; built for this output with its symjit kernels). The runner is
 * configured from the file. Nothing is read from the environment, and there is no fallback:
 * whatever the file or the runner lacks is an error at `umami_initialize`.
 *
 * Per event, as `umami.c`: the channel's flavour selects contributions (process, colour id, leg
 * permutation, factor, helicity maps); then
 *   - a fixed helicity (UMAMI_IN_HELICITY_INDEX): sum over the contributions of factor times the
 *     value of that helicity;
 *   - helicity sampling (UMAMI_OUT_HELICITY_INDEX, one contribution): the per-helicity values,
 *     the helicity drawn by UMAMI_IN_RANDOM_HELICITY in proportion to them, times the factor;
 *   - else: sum over the contributions of factor times the helicity sum.
 * The values are the recorded resolved ones (`evaluate_resolved_f64`): the sampling path is
 * RustiCol's bit for bit (f64 kernel); the other two sum the same values in RustiCol's resolved
 * order rather than its selected-path order.
 *
 * Events are batched: every event's contributions are gathered by (process, colour id, leg
 * permutation), each batch is
 * evaluated in tiles (one runner call that keeps its amplitudes, one resolved reduction), then
 * the events are combined in order. alpha_s: |M|^2(a) = |M|^2(a0) (a/a0)^p with the file's a0
 * and p (exact for a = a0); the other mutable parameters can only be set to their values at
 * conversion. */
#include "umami.h"
#include "umami_provider.h"
#include "pacman_recurrence.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef UMAMI_PROVIDER_HEADER
#define UMAMI_PROVIDER_HEADER "provider_p0.h"
#endif
#include UMAMI_PROVIDER_HEADER

/* ---- the .pacuda file (pacuda-convert's format.rs) ------------------------------------------ */

typedef struct {
    char *id;
    uint64_t key; /* the runner's cache key of the program (a hash of its words and widths) */
    uint32_t tile;
    uint32_t word_count;
    uint64_t *words;
    uint8_t width[8];
    uint32_t current_planes, amplitude_planes, momentum_planes;
    uint32_t scale_count;
    double *scale_re, *scale_im;
    uint32_t factor_count;
    double *factor_re, *factor_im;
    uint32_t zero_runs, download_runs;
    uint32_t *zero, *download;
    uint32_t external_count;
    uint32_t form_count;
    uint32_t *form_terms;
    uint32_t term_count;
    uint32_t *term_slots;
    double *term_coefficients;
    uint32_t outputs;
    uint32_t *offsets, *planes;
    double *weights;
    uint32_t reference_points;
    double *reference_momenta, *reference_values;
} PColour;

typedef struct {
    char *key;
    uint32_t legs, helicities;
    char *alpha_s_name;
    double alpha_s;
    int32_t qcd_power;
    uint32_t parameter_count;
    char **parameter_names; /* the model parameters set at conversion, and their values */
    double *parameter_re, *parameter_im;
    int64_t numerator, denominator;
    uint32_t *crossing_leg; /* NULL: identity */
    double *crossing_sign;
    uint32_t colour_count;
    PColour *colours;
} PProcess;

typedef struct {
    uint32_t kernel, batch;
    uint32_t sha_count;
    char (*sha)[65];
    uint32_t process_count;
    PProcess *processes;
} PFile;

typedef struct {
    const uint8_t *p, *end;
    int bad;
} Reader;

static int take(Reader *r, void *out, size_t bytes) {
    if (r->bad || (size_t)(r->end - r->p) < bytes) { r->bad = 1; return 0; }
    if (out) memcpy(out, r->p, bytes);
    r->p += bytes;
    return 1;
}
static uint32_t rd_u32(Reader *r) { uint32_t v = 0; take(r, &v, 4); return v; }
static int32_t rd_i32(Reader *r) { int32_t v = 0; take(r, &v, 4); return v; }
static int64_t rd_i64(Reader *r) { int64_t v = 0; take(r, &v, 8); return v; }
static double rd_f64(Reader *r) { double v = 0; take(r, &v, 8); return v; }
/* an array of n elements of `size` bytes (n checked against the bytes left) */
static void *rd_array(Reader *r, size_t n, size_t size) {
    if (r->bad || (size && n > (size_t)(r->end - r->p) / size)) { r->bad = 1; return NULL; }
    void *a = malloc(n * size + 1);
    if (!a) { r->bad = 1; return NULL; }
    take(r, a, n * size);
    return a;
}
static char *rd_str(Reader *r) {
    uint32_t n = rd_u32(r);
    char *s = rd_array(r, n, 1);
    if (s) s[n] = 0;
    return s;
}

static void free_file(PFile *f) {
    if (!f) return;
    for (uint32_t i = 0; f->processes && i < f->process_count; i++) {
        PProcess *p = &f->processes[i];
        for (uint32_t j = 0; p->colours && j < p->colour_count; j++) {
            PColour *c = &p->colours[j];
            free(c->id); free(c->words); free(c->scale_re); free(c->scale_im); free(c->factor_re);
            free(c->factor_im); free(c->zero); free(c->download); free(c->form_terms); free(c->term_slots);
            free(c->term_coefficients); free(c->offsets); free(c->planes); free(c->weights);
            free(c->reference_momenta); free(c->reference_values);
        }
        for (uint32_t k = 0; p->parameter_names && k < p->parameter_count; k++) free(p->parameter_names[k]);
        free(p->parameter_names); free(p->parameter_re); free(p->parameter_im);
        free(p->colours); free(p->key); free(p->alpha_s_name); free(p->crossing_leg); free(p->crossing_sign);
    }
    free(f->processes); free(f->sha); free(f);
}

static PFile *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "pacuda: %s: cannot open\n", path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *bytes = size > 0 ? malloc((size_t)size) : NULL;
    if (!bytes || fread(bytes, 1, (size_t)size, fp) != (size_t)size) {
        fclose(fp); free(bytes);
        fprintf(stderr, "pacuda: %s: cannot read\n", path);
        return NULL;
    }
    fclose(fp);
    Reader r = {bytes, bytes + size, 0};
    PFile *f = calloc(1, sizeof *f);
    char magic[8];
    if (!f || !take(&r, magic, 8) || memcmp(magic, "PACUDA\0\x01", 8) || rd_u32(&r) != 2) {
        fprintf(stderr, "pacuda: %s: not a version-2 .pacuda file (convert it again)\n", path);
        free(bytes); free(f);
        return NULL;
    }
    f->kernel = rd_u32(&r);
    f->batch = rd_u32(&r);
    f->sha_count = rd_u32(&r);
    if (f->sha_count > (size_t)(r.end - r.p) / 64) r.bad = 1;
    f->sha = r.bad ? NULL : calloc(f->sha_count ? f->sha_count : 1, sizeof *f->sha);
    if (!f->sha) r.bad = 1;
    for (uint32_t k = 0; f->sha && k < f->sha_count; k++) {
        take(&r, f->sha[k], 64);
        f->sha[k][64] = 0;
    }
    f->process_count = rd_u32(&r);
    f->processes = r.bad ? NULL : calloc(f->process_count ? f->process_count : 1, sizeof *f->processes);
    if (!f->processes) r.bad = 1;
    for (uint32_t i = 0; !r.bad && i < f->process_count; i++) {
        PProcess *p = &f->processes[i];
        p->key = rd_str(&r);
        p->legs = rd_u32(&r);
        p->helicities = rd_u32(&r);
        p->alpha_s_name = rd_str(&r);
        p->alpha_s = rd_f64(&r);
        p->qcd_power = rd_i32(&r);
        p->parameter_count = rd_u32(&r);
        if (p->parameter_count > (size_t)(r.end - r.p) / 20) { r.bad = 1; break; }
        p->parameter_names = calloc(p->parameter_count ? p->parameter_count : 1, sizeof(char *));
        p->parameter_re = calloc(p->parameter_count ? p->parameter_count : 1, sizeof(double));
        p->parameter_im = calloc(p->parameter_count ? p->parameter_count : 1, sizeof(double));
        if (!p->parameter_names || !p->parameter_re || !p->parameter_im) { r.bad = 1; break; }
        for (uint32_t k = 0; !r.bad && k < p->parameter_count; k++) {
            p->parameter_names[k] = rd_str(&r);
            p->parameter_re[k] = rd_f64(&r);
            p->parameter_im[k] = rd_f64(&r);
        }
        p->numerator = rd_i64(&r);
        p->denominator = rd_i64(&r);
        uint32_t crossing = rd_u32(&r);
        if (crossing) {
            if (crossing != p->legs) { r.bad = 1; break; }
            p->crossing_leg = calloc(crossing, sizeof(uint32_t));
            p->crossing_sign = calloc(crossing, sizeof(double));
            if (!p->crossing_leg || !p->crossing_sign) { r.bad = 1; break; }
            for (uint32_t l = 0; l < crossing; l++) {
                p->crossing_leg[l] = rd_u32(&r);
                p->crossing_sign[l] = rd_f64(&r);
                if (p->crossing_leg[l] >= p->legs) r.bad = 1;
            }
        }
        p->colour_count = rd_u32(&r);
        p->colours = r.bad ? NULL : calloc(p->colour_count ? p->colour_count : 1, sizeof *p->colours);
        if (!p->colours) r.bad = 1;
        for (uint32_t j = 0; !r.bad && j < p->colour_count; j++) {
            PColour *c = &p->colours[j];
            c->id = rd_str(&r);
            c->tile = rd_u32(&r);
            c->word_count = rd_u32(&r);
            c->words = rd_array(&r, c->word_count, 8);
            take(&r, c->width, 8);
            c->current_planes = rd_u32(&r);
            c->amplitude_planes = rd_u32(&r);
            c->momentum_planes = rd_u32(&r);
            c->scale_count = rd_u32(&r);
            c->scale_re = rd_array(&r, c->scale_count, 8);
            c->scale_im = rd_array(&r, c->scale_count, 8);
            c->factor_count = rd_u32(&r);
            c->factor_re = rd_array(&r, c->factor_count, 8);
            c->factor_im = rd_array(&r, c->factor_count, 8);
            c->zero_runs = rd_u32(&r);
            c->zero = rd_array(&r, 2 * (size_t)c->zero_runs, 4);
            c->download_runs = rd_u32(&r);
            c->download = rd_array(&r, 2 * (size_t)c->download_runs, 4);
            c->external_count = rd_u32(&r);
            c->form_count = rd_u32(&r);
            c->form_terms = rd_array(&r, 2 * (size_t)c->form_count, 4);
            c->term_count = rd_u32(&r);
            c->term_slots = rd_array(&r, c->term_count, 4);
            c->term_coefficients = rd_array(&r, c->term_count, 8);
            c->outputs = rd_u32(&r);
            c->offsets = rd_array(&r, (size_t)c->outputs + 1, 4);
            uint32_t count = c->offsets ? c->offsets[c->outputs] : 0;
            c->planes = rd_array(&r, count, 4);
            c->weights = rd_array(&r, count, 8);
            c->reference_points = rd_u32(&r);
            c->reference_momenta = rd_array(&r, (size_t)c->reference_points * p->legs * 4, 8);
            c->reference_values = rd_array(&r, (size_t)c->reference_points * p->helicities, 8);
            if (!r.bad && (c->outputs != p->helicities || c->external_count != p->legs || c->tile == 0 ||
                           c->offsets[0] != 0))
                r.bad = 1;
            for (uint32_t h = 0; !r.bad && h < c->outputs; h++)
                if (c->offsets[h] > c->offsets[h + 1]) r.bad = 1;
        }
    }
    if (!r.bad && r.p != r.end) r.bad = 1;
    free(bytes);
    if (r.bad) {
        fprintf(stderr, "pacuda: %s: a malformed .pacuda file\n", path);
        free_file(f);
        return NULL;
    }
    return f;
}

/* The op 14 kernel indices of `words` mapped by `map` (the file's index -> the runner's). */
static int remap_kernels(uint64_t *words, uint32_t count, const int *map, uint32_t map_count) {
    for (uint32_t n = 0; n < count;) {
        uint64_t w = words[n];
        if ((w & 15) == 14) {
            uint32_t kernel = (uint32_t)(w >> 4) & 0xffff;
            uint32_t codes = ((uint32_t)(w >> 20) & 0xff) + ((uint32_t)(w >> 28) & 0xff);
            if (kernel >= map_count) return 0;
            words[n] = (w & ~(0xffffull << 4)) | (uint64_t)map[kernel] << 4;
            n += 1 + (codes + 1) / 2;
        } else {
            n += 1;
        }
    }
    return 1;
}

/* The runner caches a program by its key: a 64-bit FNV-1a hash of the words and their widths (the
 * same program has the same key in every library of the output; 0 would mean "do not cache"). */
static uint64_t program_key(const PColour *c) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t *b = c->width;
    for (int k = 0; k < 8; k++) h = (h ^ b[k]) * 1099511628211ull;
    b = (const uint8_t *)c->words;
    for (size_t k = 0; k < (size_t)c->word_count * 8; k++) h = (h ^ b[k]) * 1099511628211ull;
    return h ? h : 1;
}

/* ---- the instance ---------------------------------------------------------------------------- */

typedef struct {
    PFile *file;
    /* per provider process: its file process; per provider process, its current alpha_s */
    PProcess **process;
    double *alpha_s;
    /* host buffers of the runner calls and the reductions */
    double *external, *values, *arena;
    size_t external_capacity, values_capacity, arena_capacity;
} Instance;

/* One evaluation: an event's contribution. */
typedef struct {
    size_t event;
    const UmamiContribution *contribution;
    double scale;     /* (alpha_s / alpha_s0)^p */
    double *values;   /* its helicity values (into the instance's value buffer) */
} Item;

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = UMAMI_HAS_ALPHA_S,
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_RANDOM_HELICITY] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
    [UMAMI_IN_CHANNEL_INDEX] = true,
};
static bool const required_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
};
static bool const supported_outputs[UMAMI_OUTPUT_KEY_COUNT] = {
    [UMAMI_OUT_MATRIX_ELEMENT] = true,
    [UMAMI_OUT_COLOR_INDEX] = UMAMI_HAS_COLOR_FLOW,
    [UMAMI_OUT_HELICITY_INDEX] = true,
};

static UmamiParameter const *parameter(size_t process, char const *name) {
    UmamiProcess const *entry = &umami_provider.processes[process];
    for (size_t i = 0; i < entry->parameter_count; ++i)
        if (strcmp(entry->parameters[i].name, name) == 0) return &entry->parameters[i];
    return NULL;
}

/* A model parameter's value at conversion: the file's (set from mg7's param card), else the
 * provider's default (the artifact's). */
static void converted_value(const PProcess *p, UmamiParameter const *entry, double *re, double *im) {
    *re = entry->real;
    *im = entry->imaginary;
    for (uint32_t k = 0; p && k < p->parameter_count; k++)
        if (strcmp(p->parameter_names[k], entry->name) == 0) {
            *re = p->parameter_re[k];
            *im = p->parameter_im[k];
        }
}

static const PColour *colour_of(const Instance *instance, const UmamiContribution *c) {
    const PProcess *p = instance->process[c->process_index];
    for (uint32_t j = 0; j < p->colour_count; j++)
        if (strcmp(p->colours[j].id, c->color_id) == 0) return &p->colours[j];
    return NULL;
}

static int grow(double **p, size_t *capacity, size_t n) {
    if (n <= *capacity) return 1;
    double *q = realloc(*p, n * sizeof(double));
    if (!q) return 0;
    *p = q;
    *capacity = n;
    return 1;
}

UmamiStatus umami_get_meta(UmamiMetaKey key, void *result) {
    if (result == NULL) return UMAMI_ERROR;
    switch (key) {
    case UMAMI_META_DEVICE: *(UmamiDevice *)result = UMAMI_DEVICE_CPU; break;
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = (int)umami_provider.particle_count; break;
    case UMAMI_META_HELICITY_COUNT: *(int *)result = (int)umami_provider.helicity_count; break;
    case UMAMI_META_COLOR_COUNT:
        if (!UMAMI_HAS_COLOR_FLOW) return UMAMI_ERROR_UNSUPPORTED_META;
        *(int *)result = (int)umami_provider.color_count; break;
    case UMAMI_META_MASSES:
        memcpy(result, umami_provider.masses, umami_provider.particle_count * sizeof(double));
        break;
    default: return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_supported_inputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_required_inputs(bool const **required, int *count) {
    if (required == NULL || count == NULL) return UMAMI_ERROR;
    *required = required_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_supported_outputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_outputs; *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
    free_file(instance->file);
    free(instance->process); free(instance->alpha_s);
    free(instance->external); free(instance->values); free(instance->arena);
    free(instance);
    return UMAMI_SUCCESS;
}

/* `param_card_path` is the LC artifact's directory (as for `umami.c`); the file is
 * `<directory>.pacuda`. */
UmamiStatus umami_initialize(UmamiHandle *handle, char const *param_card_path) {
    if (handle == NULL) return UMAMI_ERROR;
    *handle = NULL;
    struct stat info;
    if (param_card_path == NULL || param_card_path[0] != '/' || stat(param_card_path, &info) != 0 ||
        !S_ISDIR(info.st_mode) || umami_provider.process_count == 0 || umami_provider.particle_count > INT_MAX)
        return UMAMI_ERROR;
    size_t length = strlen(param_card_path);
    while (length > 1 && param_card_path[length - 1] == '/') length--;
    char *path = malloc(length + sizeof ".pacuda");
    if (!path) return UMAMI_ERROR;
    memcpy(path, param_card_path, length);
    memcpy(path + length, ".pacuda", sizeof ".pacuda");
    Instance *instance = calloc(1, sizeof *instance);
    if (!instance) { free(path); return UMAMI_ERROR; }
    instance->file = read_file(path);
    free(path);
    if (!instance->file) { umami_free(instance); return UMAMI_ERROR; }
    PFile *f = instance->file;

    /* the runner: the file's configuration, its symjit kernels */
    pacman_recurrence_options options = {PACMAN_RECURRENCE_ABI, sizeof options, f->kernel, 0, 0, 0};
    if (pacman_recurrence_v1_configure(&options) != PACMAN_RECURRENCE_OK) {
        fprintf(stderr, "pacuda: the runner refuses the configuration (kernel %u)\n", f->kernel);
        umami_free(instance);
        return UMAMI_ERROR;
    }
    int *map = calloc(f->sha_count ? f->sha_count : 1, sizeof(int));
    if (!map) { umami_free(instance); return UMAMI_ERROR; }
    for (uint32_t k = 0; k < f->sha_count; k++) {
        map[k] = pacman_recurrence_v1_symjit_kernel(f->sha[k]);
        if (map[k] < 0) {
            fprintf(stderr, "pacuda: the runner has no symjit kernel %s (rebuild it for this output)\n", f->sha[k]);
            free(map);
            umami_free(instance);
            return UMAMI_ERROR;
        }
    }
    for (uint32_t i = 0; i < f->process_count; i++)
        for (uint32_t j = 0; j < f->processes[i].colour_count; j++) {
            PColour *c = &f->processes[i].colours[j];
            int remapped = remap_kernels(c->words, c->word_count, map, f->sha_count);
            c->key = program_key(c);
            if (!remapped) {
                fprintf(stderr, "pacuda: process %s, colour %s: a symjit kernel outside the table\n",
                        f->processes[i].key, c->id);
                free(map);
                umami_free(instance);
                return UMAMI_ERROR;
            }
        }
    free(map);

    /* the provider's processes and contributions, in the file */
    instance->process = calloc(umami_provider.process_count, sizeof *instance->process);
    instance->alpha_s = calloc(umami_provider.process_count, sizeof(double));
    if (!instance->process || !instance->alpha_s) { umami_free(instance); return UMAMI_ERROR; }
    for (size_t i = 0; i < umami_provider.process_count; i++) {
        const UmamiProcess *entry = &umami_provider.processes[i];
        for (uint32_t k = 0; k < f->process_count; k++)
            if (strcmp(f->processes[k].key, entry->key) == 0) instance->process[i] = &f->processes[k];
        PProcess *p = instance->process[i];
        if (p && (p->legs != umami_provider.particle_count || p->helicities != entry->helicity_count)) {
            fprintf(stderr, "pacuda: process %s: %u legs and %u helicities in the file, %zu and %zu in the provider\n",
                    entry->key, p->legs, p->helicities, umami_provider.particle_count, entry->helicity_count);
            umami_free(instance);
            return UMAMI_ERROR;
        }
        if (p) instance->alpha_s[i] = p->alpha_s;
    }
    for (size_t c = 0; c < umami_provider.channel_count; c++)
        for (size_t fl = 0; fl < umami_provider.channels[c].flavour_count; fl++) {
            const UmamiFlavour *flavour = &umami_provider.channels[c].flavours[fl];
            for (size_t j = 0; j < flavour->contribution_count; j++) {
                const UmamiContribution *entry = &flavour->contributions[j];
                if (!instance->process[entry->process_index] || !colour_of(instance, entry)) {
                    fprintf(stderr, "pacuda: process %s, colour %s: not in the file\n",
                            umami_provider.processes[entry->process_index].key, entry->color_id);
                    umami_free(instance);
                    return UMAMI_ERROR;
                }
            }
        }
    *handle = instance;
    return UMAMI_SUCCESS;
}

UmamiStatus umami_get_parameter(UmamiHandle handle, char const *name, double *real, double *imaginary) {
    if (handle == NULL || name == NULL || real == NULL) return UMAMI_ERROR;
    Instance *instance = handle;
    for (size_t i = 0; i < umami_provider.process_count; ++i) {
        UmamiParameter const *entry = parameter(i, name);
        if (entry == NULL) continue;
        if (entry->is_complex && imaginary == NULL) return UMAMI_ERROR;
        const PProcess *p = instance->process[i];
        bool alpha = p && p->alpha_s_name[0] && strcmp(p->alpha_s_name, name) == 0;
        double re, im;
        converted_value(p, entry, &re, &im);
        *real = alpha ? instance->alpha_s[i] : re;
        if (imaginary != NULL) *imaginary = alpha ? 0.0 : im;
        return UMAMI_SUCCESS;
    }
    return UMAMI_ERROR;
}

/* alpha_s: any value (the matrix elements scale with it); another mutable parameter: only its
 * value at conversion (the file holds the couplings computed from it). */
UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name, double real, double imaginary) {
    if (handle == NULL || name == NULL || !isfinite(real)) return UMAMI_ERROR;
    Instance *instance = handle;
    bool found = false;
    for (size_t i = 0; i < umami_provider.process_count; ++i) {
        UmamiParameter const *entry = parameter(i, name);
        if (entry == NULL) continue;
        found = true;
        if (!entry->mutable || (entry->is_complex && !isfinite(imaginary))) return UMAMI_ERROR;
        const PProcess *p = instance->process[i];
        if (p && p->alpha_s_name[0] && strcmp(p->alpha_s_name, name) == 0) {
            if (!(real >= 0.0)) return UMAMI_ERROR;
            continue;
        }
        double re, im;
        converted_value(p, entry, &re, &im);
        if (real != re || (entry->is_complex && imaginary != im)) {
            fprintf(stderr, "pacuda: %s = %.17g: the file was converted with %.17g; convert it again "
                            "with this parameter card\n", name, real, re);
            return UMAMI_ERROR;
        }
    }
    if (!found) return UMAMI_ERROR;
    for (size_t i = 0; i < umami_provider.process_count; ++i) {
        const PProcess *p = instance->process[i];
        if (p && p->alpha_s_name[0] && strcmp(p->alpha_s_name, name) == 0) instance->alpha_s[i] = real;
    }
    return UMAMI_SUCCESS;
}

/* Items evaluate alike when their contributions share the process, the colour id and the leg
 * permutation (the factor, the helicity maps and the alpha_s scale are applied per item): one
 * batch for all the flavours that do (as umami.c's batched path groups them). */
static int compare_evaluation(const UmamiContribution *x, const UmamiContribution *y) {
    if (x == y) return 0;
    if (x->process_index != y->process_index) return x->process_index < y->process_index ? -1 : 1;
    int c = strcmp(x->color_id, y->color_id);
    if (c != 0) return c;
    return memcmp(x->permutation, y->permutation, umami_provider.particle_count * sizeof(size_t));
}

static int compare_items(const void *a, const void *b) {
    const Item *x = a, *y = b;
    int c = compare_evaluation(x->contribution, y->contribution);
    if (c != 0) return c;
    return x->event < y->event ? -1 : x->event > y->event;
}

/* The helicity values of `count` items of one contribution (items[k].values, each `helicities`
 * long), in tiles on the runner. */
static UmamiStatus evaluate(Instance *instance, Item *items, size_t count, double const *momenta, size_t stride) {
    const UmamiContribution *entry = items[0].contribution;
    const PProcess *p = instance->process[entry->process_index];
    const PColour *c = colour_of(instance, entry);
    size_t n = p->legs, h = p->helicities;
    size_t tile = instance->file->batch ? instance->file->batch : c->tile;
    size_t planes = (size_t)c->current_planes * 2 + (size_t)c->amplitude_planes * 2 + c->momentum_planes;
    if (!grow(&instance->external, &instance->external_capacity, n * 4 * tile) ||
        !grow(&instance->values, &instance->values_capacity, h * tile) ||
        !grow(&instance->arena, &instance->arena_capacity, planes * tile + 1))
        return UMAMI_ERROR;
    double *current_re = instance->arena, *current_im = current_re + (size_t)c->current_planes * tile;
    double *amplitude_re = current_im + (size_t)c->current_planes * tile;
    double *amplitude_im = amplitude_re + (size_t)c->amplitude_planes * tile;
    double *host_momenta = amplitude_im + (size_t)c->amplitude_planes * tile;
    for (size_t start = 0; start < count; start += tile) {
        size_t points = count - start < tile ? count - start : tile;
        /* runtime input leg j = UMAMI leg permutation[j]; external leg l = sign * input leg
         * crossing[l]; source-major planes of `points` values */
        double *external = instance->external;
        for (size_t l = 0; l < n; l++) {
            size_t source = p->crossing_leg ? p->crossing_leg[l] : l;
            double sign = p->crossing_leg ? p->crossing_sign[l] : 1.0;
            size_t leg = entry->permutation[source];
            for (size_t k = 0; k < 4; k++) {
                double *plane = external + (l * 4 + k) * points;
                for (size_t q = 0; q < points; q++) {
                    double v = momenta[items[start + q].event + stride * (n * k + leg)];
                    if (!isfinite(v)) return UMAMI_ERROR;
                    plane[q] = sign * v;
                }
            }
        }
        pacman_recurrence_call call;
        memset(&call, 0, sizeof call);
        call.abi_version = PACMAN_RECURRENCE_ABI;
        call.struct_size = sizeof call;
        call.words = c->words;
        call.words_key = c->key;
        call.word_count = c->word_count;
        memcpy(call.width, c->width, 8);
        call.current_re = current_re;
        call.current_im = current_im;
        call.amplitude_re = amplitude_re;
        call.amplitude_im = amplitude_im;
        call.momenta = host_momenta;
        call.scale_re = c->scale_re;
        call.scale_im = c->scale_im;
        call.factor_re = c->factor_re;
        call.factor_im = c->factor_im;
        call.download_amplitude = c->download;
        call.current_planes = c->current_planes;
        call.amplitude_planes = c->amplitude_planes;
        call.momentum_planes = c->momentum_planes;
        call.scale_count = c->scale_count;
        call.factor_count = c->factor_count;
        call.download_amplitude_runs = c->download_runs;
        call.stride = (uint32_t)tile;
        call.points = (uint32_t)points;
        call.zero_amplitude = c->zero;
        call.zero_amplitude_runs = c->zero_runs;
        call.flags = PACMAN_RECURRENCE_KEEP_AMPLITUDES | PACMAN_RECURRENCE_FORMS_ON_DEVICE;
        call.external_momenta = external;
        call.external_count = (uint32_t)n;
        call.term_count = c->term_count;
        call.form_terms = c->form_terms;
        call.term_slots = c->term_slots;
        call.term_coefficients = c->term_coefficients;
        if (pacman_recurrence_v1_run(&call) != PACMAN_RECURRENCE_OK) {
            fprintf(stderr, "pacuda: process %s, colour %s: the runner failed\n", p->key, c->id);
            return UMAMI_ERROR;
        }
        double *values = instance->values;
        memset(values, 0, points * h * sizeof(double));
        pacman_recurrence_reduce_resolved reduce = {PACMAN_RECURRENCE_ABI, sizeof reduce, c->planes, c->weights,
                                                    c->offsets, values, (uint32_t)h, (uint32_t)points, 1, (uint32_t)h};
        if (pacman_recurrence_v1_reduce_resolved(&reduce) != PACMAN_RECURRENCE_OK) {
            fprintf(stderr, "pacuda: process %s, colour %s: the reduction failed\n", p->key, c->id);
            return UMAMI_ERROR;
        }
        /* the process-group factor (`apply_process_group_factor`), the alpha_s scale */
        for (size_t q = 0; q < points; q++) {
            Item *item = &items[start + q];
            for (size_t hel = 0; hel < h; hel++) {
                double v = values[q * h + hel];
                if (p->numerator != p->denominator)
                    v = v * (double)p->numerator / (double)p->denominator;
                if (item->scale != 1.0) v *= item->scale;
                item->values[hel] = v;
            }
        }
    }
    return UMAMI_SUCCESS;
}

UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const *input_keys, void const *const *inputs,
    size_t output_count, UmamiOutputKey const *output_keys, void *const *outputs) {
    if (handle == NULL || offset > stride || count > stride - offset ||
        (input_count && (input_keys == NULL || inputs == NULL)) ||
        (output_count && (output_keys == NULL || outputs == NULL)) ||
        (stride && umami_provider.particle_count > SIZE_MAX / stride / 4)) return UMAMI_ERROR;
    void const *in[UMAMI_INPUT_KEY_COUNT] = {0};
    void *out[UMAMI_OUTPUT_KEY_COUNT] = {0};
    for (size_t i = 0; i < input_count; ++i) {
        int key = input_keys[i];
        if (key < 0 || key >= UMAMI_INPUT_KEY_COUNT || !supported_inputs[key])
            return UMAMI_ERROR_UNSUPPORTED_INPUT;
        if (inputs[i] == NULL || in[key] != NULL) return UMAMI_ERROR;
        in[key] = inputs[i];
    }
    for (size_t i = 0; i < output_count; ++i) {
        int key = output_keys[i];
        if (key < 0 || key >= UMAMI_OUTPUT_KEY_COUNT || !supported_outputs[key])
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
        if (outputs[i] == NULL || out[key] != NULL) return UMAMI_ERROR;
        out[key] = outputs[i];
    }
    if (count == 0) return UMAMI_SUCCESS;
    if (in[UMAMI_IN_MOMENTA] == NULL) return UMAMI_ERROR_MISSING_INPUT;
    Instance *instance = handle;
    double const *momenta = in[UMAMI_IN_MOMENTA];

    /* the items: every event's contributions (as `umami.c` would evaluate them) */
    size_t item_count = 0, value_count = 0;
    for (size_t event = offset; event < offset + count; ++event) {
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ? ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ? ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        if (channel >= umami_provider.channel_count || flavour < 0 ||
            (size_t)flavour >= umami_provider.channels[channel].flavour_count) return UMAMI_ERROR;
        UmamiFlavour const *selection = &umami_provider.channels[channel].flavours[flavour];
        if (selection->contribution_count == 0) return UMAMI_ERROR;
        double random = in[UMAMI_IN_RANDOM_HELICITY] ? ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        if (!isfinite(random) || random < 0.0 || random >= 1.0) return UMAMI_ERROR;
        if (in[UMAMI_IN_HELICITY_INDEX]) {
            int label = ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event];
            if (label < 0 || (size_t)label >= umami_provider.exported_helicity_count) return UMAMI_ERROR;
        } else if (out[UMAMI_OUT_HELICITY_INDEX] && selection->contribution_count != 1) {
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
        }
        for (size_t j = 0; j < selection->contribution_count; ++j) {
            const UmamiContribution *entry = &selection->contributions[j];
            if (in[UMAMI_IN_ALPHA_S]) {
                double a = ((double const *)in[UMAMI_IN_ALPHA_S])[event];
                if (!isfinite(a) || a < 0.0) return UMAMI_ERROR;
                if (!instance->process[entry->process_index]->alpha_s_name[0]) return UMAMI_ERROR_UNSUPPORTED_INPUT;
            }
            item_count++;
            value_count += instance->process[entry->process_index]->helicities;
        }
    }
    Item *items = malloc(item_count * sizeof *items);
    double *values = malloc((value_count ? value_count : 1) * sizeof(double));
    if (!items || !values) { free(items); free(values); return UMAMI_ERROR; }
    size_t k = 0;
    double *next = values;
    for (size_t event = offset; event < offset + count; ++event) {
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ? ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ? ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        UmamiFlavour const *selection = &umami_provider.channels[channel].flavours[flavour];
        for (size_t j = 0; j < selection->contribution_count; ++j) {
            const UmamiContribution *entry = &selection->contributions[j];
            const PProcess *p = instance->process[entry->process_index];
            double a = in[UMAMI_IN_ALPHA_S] ? ((double const *)in[UMAMI_IN_ALPHA_S])[event]
                                            : instance->alpha_s[entry->process_index];
            double scale = 1.0;
            if (p->alpha_s_name[0] && a != p->alpha_s) scale = pow(a / p->alpha_s, p->qcd_power);
            items[k] = (Item){event, entry, scale, next};
            next += p->helicities;
            k++;
        }
    }
    /* evaluate by (process, colour, permutation) (the items in event order within each) */
    Item *sorted = malloc(item_count * sizeof *sorted);
    if (!sorted) { free(items); free(values); return UMAMI_ERROR; }
    memcpy(sorted, items, item_count * sizeof *sorted);
    qsort(sorted, item_count, sizeof *sorted, compare_items);
    UmamiStatus status = UMAMI_SUCCESS;
    for (size_t first = 0; status == UMAMI_SUCCESS && first < item_count;) {
        size_t last = first;
        while (last < item_count && compare_evaluation(sorted[last].contribution, sorted[first].contribution) == 0)
            last++;
        status = evaluate(instance, sorted + first, last - first, momenta, stride);
        first = last;
    }
    free(sorted);

    /* combine the events, as `umami.c` */
    k = 0;
    for (size_t event = offset; status == UMAMI_SUCCESS && event < offset + count; ++event) {
        size_t channel = in[UMAMI_IN_CHANNEL_INDEX] ? ((unsigned int const *)in[UMAMI_IN_CHANNEL_INDEX])[event] : 0;
        int flavour = in[UMAMI_IN_FLAVOR_INDEX] ? ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0;
        UmamiFlavour const *selection = &umami_provider.channels[channel].flavours[flavour];
        const UmamiContribution *first = &selection->contributions[0];
        double random = in[UMAMI_IN_RANDOM_HELICITY] ? ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        int label = 0;
        double total = 0.0;
        if (in[UMAMI_IN_HELICITY_INDEX]) {
            label = ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event];
            for (size_t j = 0; j < selection->contribution_count; ++j) {
                const UmamiContribution *entry = &selection->contributions[j];
                const Item *item = &items[k + j];
                if (entry->helicity_runtime[label] < 0) continue;
                total += entry->factor * item->values[entry->helicity_runtime[label]];
            }
        } else if (out[UMAMI_OUT_HELICITY_INDEX]) {
            /* sample the helicity in proportion to its contribution */
            const double *v = items[k].values;
            size_t h = instance->process[first->process_index]->helicities;
            for (size_t hel = 0; hel < h; ++hel) total += v[hel];
            double target = random * total, cumulative = 0.0;
            size_t chosen = 0;
            for (size_t hel = 0; hel < h; ++hel) {
                if (v[hel] <= 0.0) continue;
                chosen = hel;
                cumulative += v[hel];
                if (cumulative > target) break;
            }
            label = (int)first->helicity_export[chosen];
            total *= first->factor;
        } else {
            for (size_t j = 0; j < selection->contribution_count; ++j) {
                const UmamiContribution *entry = &selection->contributions[j];
                const double *v = items[k + j].values;
                size_t h = instance->process[entry->process_index]->helicities;
                double value = 0.0;
                for (size_t hel = 0; hel < h; ++hel) value += v[hel];
                total += entry->factor * value;
            }
        }
        k += selection->contribution_count;
        if (!isfinite(total)) { status = UMAMI_ERROR; break; }
        if (out[UMAMI_OUT_MATRIX_ELEMENT]) ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[event] = total;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = label;
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] = (int)first->color_index;
    }
    free(items);
    free(values);
    return status;
}
