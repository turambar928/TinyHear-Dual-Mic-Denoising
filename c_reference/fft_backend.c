#include "fft_backend.h"

#include <math.h>

#include "generated/band_matrix.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void tiny_rfft_forward(const float *time_data, TinyComplex32 *freq_data) {
    for (int k = 0; k < TINY_TCN_FREQ_BINS; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (int n = 0; n < TINY_TCN_N_FFT; ++n) {
            double angle = -2.0 * M_PI * (double)k * (double)n / (double)TINY_TCN_N_FFT;
            re += (double)time_data[n] * cos(angle);
            im += (double)time_data[n] * sin(angle);
        }
        freq_data[k].re = (float)re;
        freq_data[k].im = (float)im;
    }
}

void tiny_rfft_inverse(const TinyComplex32 *freq_data, float *time_data) {
    for (int n = 0; n < TINY_TCN_N_FFT; ++n) {
        double sum = (double)freq_data[0].re;
        sum += (double)freq_data[TINY_TCN_FREQ_BINS - 1].re * ((n % 2 == 0) ? 1.0 : -1.0);
        for (int k = 1; k < TINY_TCN_FREQ_BINS - 1; ++k) {
            double angle = 2.0 * M_PI * (double)k * (double)n / (double)TINY_TCN_N_FFT;
            sum += 2.0 * ((double)freq_data[k].re * cos(angle) - (double)freq_data[k].im * sin(angle));
        }
        time_data[n] = (float)(sum / (double)TINY_TCN_N_FFT);
    }
}
