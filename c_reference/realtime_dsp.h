#pragma once

#include "tiny_tcn_int8.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float input_buffer[2][TINY_TCN_N_FFT];
    float output_buffer[TINY_TCN_N_FFT];
    float norm_buffer[TINY_TCN_N_FFT];
    TinyTcnState model_state;
} TinyRealtimeDspState;

void tiny_realtime_init(TinyRealtimeDspState *state);
void tiny_realtime_process_hop(TinyRealtimeDspState *state, const float *stereo_hop, float *output_hop);

#ifdef __cplusplus
}
#endif
