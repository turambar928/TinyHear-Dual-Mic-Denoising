#pragma once

#include "generated/model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float sum[TINY_TCN_FEATURE_DIM];
    float sumsq[TINY_TCN_FEATURE_DIM];
    int frames;
} TinyGateState;

void tiny_gate_init(TinyGateState *state);
void tiny_gate_update(TinyGateState *state, const float *features);
float tiny_gate_compute_from_stats(const TinyGateState *state);
float tiny_gate_compute_from_pooled(const float *pooled);

#ifdef __cplusplus
}
#endif
