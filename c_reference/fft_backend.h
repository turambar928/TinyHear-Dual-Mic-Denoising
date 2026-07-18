#pragma once

#include "generated/model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float re;
    float im;
} TinyComplex32;

void tiny_rfft_forward(const float *time_data, TinyComplex32 *freq_data);
void tiny_rfft_inverse(const TinyComplex32 *freq_data, float *time_data);

#ifdef __cplusplus
}
#endif
