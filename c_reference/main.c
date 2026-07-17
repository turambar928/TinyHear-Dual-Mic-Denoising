#include <math.h>
#include <stdio.h>

#include "generated/test_vectors.h"
#include "tiny_tcn_int8.h"

int main(void) {
    float output[TEST_OUTPUT_SIZE];
    int16_t output_q15[TEST_OUTPUT_SIZE];
    tiny_tcn_forward(kTestInput, TEST_FRAMES, output);
    tiny_tcn_forward_q15(kTestInput, TEST_FRAMES, output_q15);

    double abs_sum = 0.0;
    float max_abs = 0.0f;
    double int_abs_sum = 0.0;
    float int_max_abs = 0.0f;
    for (int i = 0; i < TEST_OUTPUT_SIZE; ++i) {
        float diff = fabsf(output[i] - kExpectedOutput[i]);
        abs_sum += diff;
        if (diff > max_abs) max_abs = diff;
        float int_out = (float)output_q15[i] / 32767.0f;
        float int_diff = fabsf(int_out - kExpectedOutput[i]);
        int_abs_sum += int_diff;
        if (int_diff > int_max_abs) int_max_abs = int_diff;
    }
    printf("frames=%d\n", TEST_FRAMES);
    printf("mixed_scale_max_abs_diff=%.9f\n", max_abs);
    printf("mixed_scale_mean_abs_diff=%.9f\n", (float)(abs_sum / TEST_OUTPUT_SIZE));
    printf("integer_max_abs_diff=%.9f\n", int_max_abs);
    printf("integer_mean_abs_diff=%.9f\n", (float)(int_abs_sum / TEST_OUTPUT_SIZE));
    return int_max_abs < 0.08f ? 0 : 1;
}
