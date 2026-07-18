#pragma once

#include <stdint.h>

#include "generated/model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TINY_TCN_MAX_LEFT_PAD ((TINY_TCN_KERNEL - 1) * 8)

typedef struct {
    int8_t block_history[TINY_TCN_BLOCKS][TINY_TCN_CHANNELS][TINY_TCN_MAX_LEFT_PAD];
    int frames_seen;
} TinyTcnState;

void tiny_tcn_init(TinyTcnState *state);
void tiny_tcn_forward(const float *input, int frames, float *output);
void tiny_tcn_forward_q15(const float *input, int frames, int16_t *output_q15);
void tiny_tcn_process_frame_q15(TinyTcnState *state, const float *input_frame, int16_t *output_q15);

#ifdef __cplusplus
}
#endif
