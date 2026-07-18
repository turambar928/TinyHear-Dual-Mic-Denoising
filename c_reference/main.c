#include <math.h>
#include <stdio.h>

#include "generated/realtime_vectors.h"
#include "generated/test_vectors.h"
#include "realtime_dsp.h"
#include "tiny_tcn_int8.h"

int main(void) {
    float output[TEST_OUTPUT_SIZE];
    int16_t output_q15[TEST_OUTPUT_SIZE];
    int16_t stream_q15[TEST_OUTPUT_SIZE];
    TinyTcnState state;
    tiny_tcn_forward(kTestInput, TEST_FRAMES, output);
    tiny_tcn_forward_q15(kTestInput, TEST_FRAMES, output_q15);
    tiny_tcn_init(&state);
    for (int t = 0; t < TEST_FRAMES; ++t) {
        float frame[TINY_TCN_FEATURE_DIM];
        int16_t frame_out[TINY_TCN_BANDS];
        for (int ch = 0; ch < TINY_TCN_FEATURE_DIM; ++ch) {
            frame[ch] = kTestInput[ch * TEST_FRAMES + t];
        }
        tiny_tcn_process_frame_q15(&state, frame, frame_out);
        for (int band = 0; band < TINY_TCN_BANDS; ++band) {
            stream_q15[band * TEST_FRAMES + t] = frame_out[band];
        }
    }

    double abs_sum = 0.0;
    float max_abs = 0.0f;
    double int_abs_sum = 0.0;
    float int_max_abs = 0.0f;
    int stream_mismatches = 0;
    for (int i = 0; i < TEST_OUTPUT_SIZE; ++i) {
        float diff = fabsf(output[i] - kExpectedOutput[i]);
        abs_sum += diff;
        if (diff > max_abs) max_abs = diff;
        float int_out = (float)output_q15[i] / 32767.0f;
        float int_diff = fabsf(int_out - kExpectedOutput[i]);
        int_abs_sum += int_diff;
        if (int_diff > int_max_abs) int_max_abs = int_diff;
        if (stream_q15[i] != output_q15[i]) stream_mismatches += 1;
    }
    printf("frames=%d\n", TEST_FRAMES);
    printf("mixed_scale_max_abs_diff=%.9f\n", max_abs);
    printf("mixed_scale_mean_abs_diff=%.9f\n", (float)(abs_sum / TEST_OUTPUT_SIZE));
    printf("integer_max_abs_diff=%.9f\n", int_max_abs);
    printf("integer_mean_abs_diff=%.9f\n", (float)(int_abs_sum / TEST_OUTPUT_SIZE));
    printf("stream_mismatches=%d\n", stream_mismatches);

    TinyRealtimeDspState dsp_state;
    float realtime_output[REALTIME_OUTPUT_SAMPLES];
    tiny_realtime_init(&dsp_state);
    for (int hop = 0; hop < REALTIME_TOTAL_HOPS; ++hop) {
        float stereo_hop[2 * TINY_TCN_HOP_LENGTH];
        float output_hop[TINY_TCN_HOP_LENGTH];
        for (int ch = 0; ch < 2; ++ch) {
            for (int i = 0; i < TINY_TCN_HOP_LENGTH; ++i) {
                int sample = hop * TINY_TCN_HOP_LENGTH + i;
                if (sample < REALTIME_INPUT_SAMPLES) {
                    stereo_hop[ch * TINY_TCN_HOP_LENGTH + i] =
                        kRealtimeInput[ch * REALTIME_INPUT_SAMPLES + sample];
                } else {
                    stereo_hop[ch * TINY_TCN_HOP_LENGTH + i] = 0.0f;
                }
            }
        }
        tiny_realtime_process_hop(&dsp_state, stereo_hop, output_hop);
        for (int i = 0; i < TINY_TCN_HOP_LENGTH; ++i) {
            realtime_output[hop * TINY_TCN_HOP_LENGTH + i] = output_hop[i];
        }
    }

    double realtime_abs_sum = 0.0;
    float realtime_max_abs = 0.0f;
    for (int i = 0; i < REALTIME_OUTPUT_SAMPLES; ++i) {
        float diff = fabsf(realtime_output[i] - kRealtimeExpectedOutput[i]);
        realtime_abs_sum += diff;
        if (diff > realtime_max_abs) realtime_max_abs = diff;
    }
    float realtime_mean_abs = (float)(realtime_abs_sum / REALTIME_OUTPUT_SAMPLES);
    printf("realtime_dsp_max_abs_diff=%.9f\n", realtime_max_abs);
    printf("realtime_dsp_mean_abs_diff=%.9f\n", realtime_mean_abs);

    return int_max_abs < 0.14f &&
                   (float)(int_abs_sum / TEST_OUTPUT_SIZE) < 0.02f &&
                   stream_mismatches == 0 &&
                   realtime_max_abs < 0.45f &&
                   realtime_mean_abs < 0.002f
               ? 0
               : 1;
}
