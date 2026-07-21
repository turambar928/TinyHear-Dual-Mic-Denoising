#include "tiny_gate.h"

#include <math.h>
#include <string.h>

#include "generated/tiny_gate_params.h"

void tiny_gate_init(TinyGateState *state) {
    memset(state, 0, sizeof(*state));
}

void tiny_gate_update(TinyGateState *state, const float *features) {
    for (int i = 0; i < TINY_TCN_FEATURE_DIM; ++i) {
        state->sum[i] += features[i];
        state->sumsq[i] += features[i] * features[i];
    }
    state->frames += 1;
}

float tiny_gate_compute_from_pooled(const float *pooled) {
    float hidden[TINY_GATE_HIDDEN];
    for (int h = 0; h < TINY_GATE_HIDDEN; ++h) {
        float acc = kTinyGateFc1Bias[h];
        for (int i = 0; i < TINY_GATE_INPUT_DIM; ++i) {
            acc += kTinyGateFc1Weight[h][i] * pooled[i];
        }
        hidden[h] = acc > 0.0f ? acc : 0.0f;
    }

    float logit = kTinyGateFc2Bias;
    for (int h = 0; h < TINY_GATE_HIDDEN; ++h) {
        logit += kTinyGateFc2Weight[h] * hidden[h];
    }
    if (logit >= 0.0f) {
        float z = expf(-logit);
        return 1.0f / (1.0f + z);
    }
    float z = expf(logit);
    return z / (1.0f + z);
}

float tiny_gate_compute_from_stats(const TinyGateState *state) {
    float pooled[TINY_GATE_INPUT_DIM];
    int frames = state->frames > 0 ? state->frames : 1;
    float denom = (float)frames;
    for (int i = 0; i < TINY_TCN_FEATURE_DIM; ++i) {
        float mean = state->sum[i] / denom;
        float var = state->sumsq[i] / denom - mean * mean;
        if (var < 1.0e-8f) var = 1.0e-8f;
        pooled[i] = mean;
        pooled[TINY_TCN_FEATURE_DIM + i] = sqrtf(var);
    }
    return tiny_gate_compute_from_pooled(pooled);
}
