#include "realtime_dsp.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "fft_backend.h"
#include "generated/band_matrix.h"
#include "generated/model_config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float clampf_local(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float hann(int n) {
    return 0.5f - 0.5f * cosf((float)(2.0 * M_PI * n) / (float)TINY_TCN_N_FFT);
}

static void make_features(const TinyComplex32 *spec0, const TinyComplex32 *spec1, float *features) {
    for (int band = 0; band < TINY_TCN_BANDS; ++band) {
        double p0 = 0.0;
        double p1 = 0.0;
        for (int bin = 0; bin < TINY_TCN_FREQ_BINS; ++bin) {
            double e0 = (double)spec0[bin].re * (double)spec0[bin].re + (double)spec0[bin].im * (double)spec0[bin].im;
            double e1 = (double)spec1[bin].re * (double)spec1[bin].re + (double)spec1[bin].im * (double)spec1[bin].im;
            p0 += e0 * (double)kBandMatrix[bin][band];
            p1 += e1 * (double)kBandMatrix[bin][band];
        }
        float p0f = (float)(p0 > 1e-8 ? p0 : 1e-8);
        float p1f = (float)(p1 > 1e-8 ? p1 : 1e-8);
        features[band] = clampf_local(logf(p0f), -20.0f, 20.0f);
        features[TINY_TCN_BANDS + band] = clampf_local(logf(p1f), -20.0f, 20.0f);
        features[2 * TINY_TCN_BANDS + band] = clampf_local(logf(p0f / p1f), -20.0f, 20.0f);
    }
}

static void mask_to_bins(const int16_t *band_mask_q15, float *bin_mask) {
    for (int bin = 0; bin < TINY_TCN_FREQ_BINS; ++bin) {
        double mask = 0.0;
        for (int band = 0; band < TINY_TCN_BANDS; ++band) {
            mask += ((double)band_mask_q15[band] / 32767.0) * (double)kBandMatrix[bin][band];
        }
        bin_mask[bin] = clampf_local((float)mask, 0.0f, 1.0f);
    }
}

void tiny_realtime_init(TinyRealtimeDspState *state) {
    memset(state, 0, sizeof(*state));
    tiny_tcn_init(&state->model_state);
}

void tiny_realtime_process_hop(TinyRealtimeDspState *state, const float *stereo_hop, float *output_hop) {
    float frame0[TINY_TCN_N_FFT];
    float frame1[TINY_TCN_N_FFT];
    float enhanced_frame[TINY_TCN_N_FFT];
    float features[TINY_TCN_FEATURE_DIM];
    float bin_mask[TINY_TCN_FREQ_BINS];
    int16_t band_mask_q15[TINY_TCN_BANDS];
    TinyComplex32 spec0[TINY_TCN_FREQ_BINS];
    TinyComplex32 spec1[TINY_TCN_FREQ_BINS];

    for (int ch = 0; ch < 2; ++ch) {
        memmove(state->input_buffer[ch], state->input_buffer[ch] + TINY_TCN_HOP_LENGTH,
                (TINY_TCN_N_FFT - TINY_TCN_HOP_LENGTH) * sizeof(float));
        for (int i = 0; i < TINY_TCN_HOP_LENGTH; ++i) {
            state->input_buffer[ch][TINY_TCN_N_FFT - TINY_TCN_HOP_LENGTH + i] =
                stereo_hop[ch * TINY_TCN_HOP_LENGTH + i];
        }
    }

    for (int n = 0; n < TINY_TCN_N_FFT; ++n) {
        float w = hann(n);
        frame0[n] = state->input_buffer[0][n] * w;
        frame1[n] = state->input_buffer[1][n] * w;
    }
    tiny_rfft_forward(frame0, spec0);
    tiny_rfft_forward(frame1, spec1);
    make_features(spec0, spec1, features);
    tiny_tcn_process_frame_q15(&state->model_state, features, band_mask_q15);
    mask_to_bins(band_mask_q15, bin_mask);
    for (int bin = 0; bin < TINY_TCN_FREQ_BINS; ++bin) {
        spec0[bin].re *= bin_mask[bin];
        spec0[bin].im *= bin_mask[bin];
    }
    tiny_rfft_inverse(spec0, enhanced_frame);

    for (int n = 0; n < TINY_TCN_N_FFT; ++n) {
        float w = hann(n);
        state->output_buffer[n] += enhanced_frame[n] * w;
        state->norm_buffer[n] += w * w;
    }
    for (int i = 0; i < TINY_TCN_HOP_LENGTH; ++i) {
        if (state->norm_buffer[i] > 1e-6f) {
            output_hop[i] = state->output_buffer[i] / state->norm_buffer[i];
        } else {
            output_hop[i] = 0.0f;
        }
    }
    memmove(state->output_buffer, state->output_buffer + TINY_TCN_HOP_LENGTH,
            (TINY_TCN_N_FFT - TINY_TCN_HOP_LENGTH) * sizeof(float));
    memmove(state->norm_buffer, state->norm_buffer + TINY_TCN_HOP_LENGTH,
            (TINY_TCN_N_FFT - TINY_TCN_HOP_LENGTH) * sizeof(float));
    memset(state->output_buffer + TINY_TCN_N_FFT - TINY_TCN_HOP_LENGTH, 0,
           TINY_TCN_HOP_LENGTH * sizeof(float));
    memset(state->norm_buffer + TINY_TCN_N_FFT - TINY_TCN_HOP_LENGTH, 0,
           TINY_TCN_HOP_LENGTH * sizeof(float));
}
