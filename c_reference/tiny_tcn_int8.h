#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void tiny_tcn_forward(const float *input, int frames, float *output);
void tiny_tcn_forward_q15(const float *input, int frames, int16_t *output_q15);

#ifdef __cplusplus
}
#endif
