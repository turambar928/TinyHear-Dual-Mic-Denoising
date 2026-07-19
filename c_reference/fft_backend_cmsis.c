#include "fft_backend.h"

#include <string.h>

#include "arm_math.h"
#include "generated/band_matrix.h"

static arm_rfft_fast_instance_f32 g_rfft;
static int g_rfft_ready = 0;

static void ensure_rfft_ready(void) {
    if (!g_rfft_ready) {
        arm_rfft_fast_init_f32(&g_rfft, TINY_TCN_N_FFT);
        g_rfft_ready = 1;
    }
}

void tiny_rfft_forward(const float *time_data, TinyComplex32 *freq_data) {
    float packed[TINY_TCN_N_FFT];
    ensure_rfft_ready();
    arm_rfft_fast_f32(&g_rfft, (float *)time_data, packed, 0);
    freq_data[0].re = packed[0];
    freq_data[0].im = 0.0f;
    freq_data[TINY_TCN_FREQ_BINS - 1].re = packed[1];
    freq_data[TINY_TCN_FREQ_BINS - 1].im = 0.0f;
    for (int k = 1; k < TINY_TCN_FREQ_BINS - 1; ++k) {
        freq_data[k].re = packed[2 * k];
        freq_data[k].im = packed[2 * k + 1];
    }
}

void tiny_rfft_inverse(const TinyComplex32 *freq_data, float *time_data) {
    float packed[TINY_TCN_N_FFT];
    memset(packed, 0, sizeof(packed));
    ensure_rfft_ready();
    packed[0] = freq_data[0].re;
    packed[1] = freq_data[TINY_TCN_FREQ_BINS - 1].re;
    for (int k = 1; k < TINY_TCN_FREQ_BINS - 1; ++k) {
        packed[2 * k] = freq_data[k].re;
        packed[2 * k + 1] = freq_data[k].im;
    }
    arm_rfft_fast_f32(&g_rfft, packed, time_data, 1);
}
