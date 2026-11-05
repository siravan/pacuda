#include <dlfcn.h>
#include "pacman_recurrence.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t abi_version, struct_size;
    const uint64_t *words;
    uint64_t words_key;
    uint32_t word_count;
    uint8_t width[8];
    double *current_re, *current_im, *amplitude_re, *amplitude_im;
    const double *momenta, *scale_re, *scale_im, *factor_re, *factor_im;
    const uint32_t *upload_current, *upload_amplitude, *download_amplitude, *download_current;
    uint32_t current_planes, amplitude_planes, momentum_planes, scale_count, factor_count;
    uint32_t upload_current_runs, upload_amplitude_runs, download_amplitude_runs,
        download_current_runs;
    uint32_t stride, points;
    const uint32_t *zero_amplitude;
    uint32_t zero_amplitude_runs, flags;
} call_t;

typedef struct {
    uint32_t abi_version, struct_size;
    const uint32_t *planes;
    const double *weights;
    double *output;
    uint32_t count, points;
} reduce_t;

enum { P = 12, A = 9, F = 6, X = 3, S = 3, K = 4 };
static uint64_t words[64];
static uint32_t nwords;
static uint64_t w_;
static int at_;
static void put(uint64_t v, int bits) { w_ |= v << at_; at_ += bits; }
static void begin(int op) { w_ = 0; at_ = 0; put(op, 4); }
static void end(void) { words[nwords++] = w_; }
static void clear(int base, int count) { begin(1); put(base, P); put(count, P); end(); }
static void contrib(int op, int init, int slot, int p0, int p1, int dst, int f0, int f1, int x) {
    begin(op); put(init, 1); put(slot, S); put(p0, P); put(p1, P); put(dst, P);
    put(f0, F); put(f1, F); put(x, X); end();
}
static void prop(int op, int base, int form, int x) { begin(op); put(base, P); put(form, F); put(x, X); end(); }
static void identity(int base, int count, int x) { begin(10); put(base, P); put(count, K); put(x, X); end(); }
static void closure(int p0, int p1, int amp, int x, int first, int count) {
    begin(11); put(p0, P); put(p1, P); put(amp, A); put(x, X); put(first, X); put(count, K); end();
}

int main(int argc, char **argv) {
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    int (*run)(const call_t *) = (int (*)(const call_t *))dlsym(lib, "pacman_recurrence_v1_run");
    int (*reduce)(const reduce_t *) = (int (*)(const reduce_t *))dlsym(lib, "pacman_recurrence_v1_reduce");
    if (!run || !reduce) return 1;
    /* rectest LIB [f64|df64|mixed|mixed64] [nosort]: the runner's configuration */
    int (*configure)(const pacman_recurrence_options *) =
        (int (*)(const pacman_recurrence_options *))dlsym(lib, "pacman_recurrence_v1_configure");
    if (argc > 2 && configure) {
        pacman_recurrence_options o = {PACMAN_RECURRENCE_ABI, sizeof o, PACMAN_RECURRENCE_KERNEL_DF64, 0, 0, 0};
        if (!strcmp(argv[2], "f64")) o.kernel = PACMAN_RECURRENCE_KERNEL_F64;
        else if (!strcmp(argv[2], "mixed")) o.kernel = PACMAN_RECURRENCE_KERNEL_MIXED;
        else if (!strcmp(argv[2], "mixed64")) o.kernel = PACMAN_RECURRENCE_KERNEL_MIXED64;
        else if (strcmp(argv[2], "df64")) { fprintf(stderr, "unknown kernel %s\n", argv[2]); return 2; }
        o.no_sort = argc > 3 && !strcmp(argv[3], "nosort");
        if (configure(&o) != 0) { fprintf(stderr, "configure failed\n"); return 1; }
    }
    /* sources: Weyl 0, 2; vectors 4, 8, 12; tensor 16 (22 planes); stage 1 at 32..43, stage 2 at 44..49 */
    clear(32, 12);
    contrib(2, 1, 0, 0, 4, 32, 0, 0, 1);
    contrib(4, 1, 1, 16, 4, 34, 0, 0, 2);
    contrib(5, 1, 2, 4, 8, 38, 0, 0, 0);
    contrib(3, 0, 1, 2, 8, 32, 0, 0, 3);
    contrib(6, 0, 0, 4, 8, 34, 0, 1, 4);
    contrib(5, 0, 1, 8, 12, 38, 0, 0, 5);
    contrib(2, 0, 2, 0, 12, 32, 0, 0, 0);
    contrib(4, 0, 0, 16, 12, 34, 0, 0, 1);
    prop(7, 32, 2, 1);
    prop(9, 34, 3, 2);
    identity(38, 6, 3);
    clear(44, 6);
    contrib(3, 1, 0, 32, 34, 44, 0, 0, 1);
    contrib(4, 1, 1, 38, 34, 46, 0, 0, 2);
    contrib(2, 0, 2, 0, 34, 44, 0, 0, 3);
    contrib(6, 0, 1, 34, 4, 46, 1, 2, 4);
    prop(8, 44, 1, 5);
    closure(44, 0, 0, 0, 2, 2);
    closure(46, 8, 1, 1, 3, 2);
    closure(32, 2, 2, 2, 0, 2);
    enum { POINTS = 300, STRIDE = 320, CP = 50, AP = 3, MP = 16 };
    static double cre[CP * STRIDE], cim[CP * STRIDE], are[AP * STRIDE], aim[AP * STRIDE], mom[MP * STRIDE];
    for (int k = 0; k < CP * STRIDE; k++) {
        cre[k] = sin(0.37 * k + 0.1);
        cim[k] = cos(0.53 * k + 0.2);
    }
    for (int f = 0; f < 4; f++)
        for (int i = 0; i < STRIDE; i++)
            for (int mu = 0; mu < 4; mu++)
                mom[(f * 4 + mu) * STRIDE + i] = mu == 0 ? 2.5 + 0.4 * sin(0.7 * i + f) : sin(0.3 * i + mu + 2 * f);
    double sre[3] = {0.5, -1.25, 0.75}, sim[3] = {0.25, 0.5, -0.125};
    double fre[6] = {1.0, -0.5, 0.25, 2.0, -1.0, 0.75}, fim[6] = {0.0, 0.5, -0.75, 0.0, 0.25, -0.5};
    uint32_t up_c[2] = {0, 22}, up_a[2] = {0, AP}, down_a[2] = {0, AP};
    call_t c;
    memset(&c, 0, sizeof c);
    c.abi_version = 1;
    c.struct_size = (uint32_t)offsetof(call_t, zero_amplitude); /* the struct before the reduction */
    c.words = words;
    c.words_key = 7;
    c.word_count = nwords;
    uint8_t width[8] = {P, A, F, X, S, K, 0, 0};
    memcpy(c.width, width, 8);
    c.current_re = cre; c.current_im = cim; c.amplitude_re = are; c.amplitude_im = aim;
    c.momenta = mom; c.scale_re = sre; c.scale_im = sim; c.factor_re = fre; c.factor_im = fim;
    c.upload_current = up_c; c.upload_amplitude = up_a; c.download_amplitude = down_a;
    c.current_planes = CP; c.amplitude_planes = AP; c.momentum_planes = MP; c.scale_count = 3;
    c.factor_count = 6; c.upload_current_runs = 1; c.upload_amplitude_runs = 1;
    c.download_amplitude_runs = 1; c.stride = STRIDE; c.points = POINTS;
    /* twice: the second call reuses the cached program and the device arena */
    for (int r = 0; r < 2; r++) {
        memset(are, 0, sizeof are);
        memset(aim, 0, sizeof aim);
        int status = run(&c);
        if (status == 3) return 77; /* device error: no usable GPU */
        if (status != 0) { fprintf(stderr, "status %d\n", status); return 1; }
    }
    /* the amplitudes cleared on the device and kept there, then reduced: as the host reduces them */
    static double want[POINTS], got[POINTS];
    uint32_t planes[4] = {2, 0, 1, 0};
    double weights[4] = {0.75, 1.5, -0.25, 2.0};
    for (int i = 0; i < POINTS; i++) {
        want[i] = got[i] = 0.125 * i;
        for (int k = 0; k < 4; k++) {
            double re = are[planes[k] * STRIDE + i], im = aim[planes[k] * STRIDE + i];
            want[i] += weights[k] * fma(re, re, im * im);
        }
    }
    static double are2[AP * STRIDE], aim2[AP * STRIDE];
    for (int k = 0; k < AP * STRIDE; k++) are2[k] = aim2[k] = 1e300; /* must not be uploaded */
    call_t k2 = c;
    k2.struct_size = sizeof k2;
    k2.amplitude_re = are2;
    k2.amplitude_im = aim2;
    k2.upload_amplitude_runs = 0;
    k2.zero_amplitude = up_a;
    k2.zero_amplitude_runs = 1;
    k2.flags = 1;
    if (run(&k2) != 0) return 1;
    for (int k = 0; k < AP * STRIDE; k++)
        if (are2[k] != 1e300 || aim2[k] != 1e300) { fprintf(stderr, "kept amplitudes were copied back\n"); return 1; }
    reduce_t r = {1, sizeof r, planes, weights, got, 4, POINTS};
    if (reduce(&r) != 0) { fprintf(stderr, "reduce failed\n"); return 1; }
    for (int i = 0; i < POINTS; i++)
        if (memcmp(&want[i], &got[i], 8)) {
            fprintf(stderr, "reduction, point %d: %.17g for %.17g\n", i, got[i], want[i]);
            return 1;
        }
    /* the resolved reduction: a kept run reduced into 3 outputs ([point][output], output 1 empty,
     * plane 0 in both others), as the host reduces each output's planes in order */
    int (*reduce_resolved)(const pacman_recurrence_reduce_resolved *) =
        (int (*)(const pacman_recurrence_reduce_resolved *))dlsym(lib, "pacman_recurrence_v1_reduce_resolved");
    if (!reduce_resolved) { fprintf(stderr, "no pacman_recurrence_v1_reduce_resolved\n"); return 1; }
    {
        enum { OUTS = 3 };
        uint32_t rplanes[5] = {2, 0, 0, 1, 0};
        double rweights[5] = {0.75, 1.5, 2.0, -0.25, 0.5};
        uint32_t offsets[OUTS + 1] = {0, 2, 2, 5};
        static double rwant[POINTS * OUTS], rgot[POINTS * OUTS];
        for (int i = 0; i < POINTS; i++)
            for (int o = 0; o < OUTS; o++) {
                double v = 0.25 * i - o;
                for (uint32_t k = offsets[o]; k < offsets[o + 1]; k++) {
                    double re = are[rplanes[k] * STRIDE + i], im = aim[rplanes[k] * STRIDE + i];
                    v += rweights[k] * fma(re, re, im * im);
                }
                rwant[i * OUTS + o] = v;
                rgot[i * OUTS + o] = 0.25 * i - o;
            }
        if (run(&k2) != 0) return 1;
        pacman_recurrence_reduce_resolved rr = {1, sizeof rr, rplanes, rweights, offsets, rgot, OUTS, POINTS, 1, OUTS, 0};
        if (reduce_resolved(&rr) != 0) { fprintf(stderr, "reduce_resolved failed\n"); return 1; }
        for (int j = 0; j < POINTS * OUTS; j++)
            if (memcmp(&rwant[j], &rgot[j], 8)) {
                fprintf(stderr, "resolved reduction, value %d: %.17g for %.17g\n", j, rgot[j], rwant[j]);
                return 1;
            }
        /* again from zero (PACMAN_RECURRENCE_OUTPUT_ZERO): what is in the output is not read */
        for (int i = 0; i < POINTS; i++)
            for (int o = 0; o < OUTS; o++) {
                double v = 0.0;
                for (uint32_t k = offsets[o]; k < offsets[o + 1]; k++) {
                    double re = are[rplanes[k] * STRIDE + i], im = aim[rplanes[k] * STRIDE + i];
                    v += rweights[k] * fma(re, re, im * im);
                }
                rwant[i * OUTS + o] = v;
                rgot[i * OUTS + o] = 1e300;
            }
        if (run(&k2) != 0) return 1;
        rr.flags = PACMAN_RECURRENCE_OUTPUT_ZERO;
        if (reduce_resolved(&rr) != 0) { fprintf(stderr, "reduce_resolved (zero) failed\n"); return 1; }
        for (int j = 0; j < POINTS * OUTS; j++)
            if (memcmp(&rwant[j], &rgot[j], 8)) {
                fprintf(stderr, "resolved reduction from zero, value %d: %.17g for %.17g\n", j, rgot[j], rwant[j]);
                return 1;
            }
    }
    for (int p = 0; p < AP; p++)
        for (int i = 0; i < POINTS; i++) printf("%.17g %.17g\n", are[p * STRIDE + i], aim[p * STRIDE + i]);
    return 0;
}
