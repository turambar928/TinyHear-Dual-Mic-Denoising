#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "generated/realtime_vectors.h"
#include "generated/test_vectors.h"
#include "realtime_dsp.h"
#include "tiny_tcn_int8.h"

static double elapsed_ms(clock_t start, clock_t end) {
    return 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC;
}

int main(void) {
    enum { MODEL_LOOPS = 200, DSP_LOOPS = 20 };
    volatile int32_t checksum = 0;

    clock_t model_start = clock();
    for (int loop = 0; loop < MODEL_LOOPS; ++loop) {
        TinyTcnState state;
        tiny_tcn_init(&state);
        for (int t = 0; t < TEST_FRAMES; ++t) {
            float frame[TINY_TCN_FEATURE_DIM];
            int16_t output[TINY_TCN_BANDS];
            for (int ch = 0; ch < TINY_TCN_FEATURE_DIM; ++ch) {
                frame[ch] = kTestInput[ch * TEST_FRAMES + t];
            }
            tiny_tcn_process_frame_q15(&state, frame, output);
            checksum += output[0];
        }
    }
    clock_t model_end = clock();

    clock_t dsp_start = clock();
    for (int loop = 0; loop < DSP_LOOPS; ++loop) {
        TinyRealtimeDspState state;
        tiny_realtime_init(&state);
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
            tiny_realtime_process_hop(&state, stereo_hop, output_hop);
            checksum += (int32_t)(output_hop[0] * 32768.0f);
        }
    }
    clock_t dsp_end = clock();

    int model_frames = MODEL_LOOPS * TEST_FRAMES;
    int dsp_hops = DSP_LOOPS * REALTIME_TOTAL_HOPS;
    double model_total_ms = elapsed_ms(model_start, model_end);
    double dsp_total_ms = elapsed_ms(dsp_start, dsp_end);
    double model_ms_per_frame = model_total_ms / (double)model_frames;
    double dsp_ms_per_hop = dsp_total_ms / (double)dsp_hops;

    printf("checksum=%d\n", (int)checksum);
    printf("model_frames=%d\n", model_frames);
    printf("model_total_ms=%.3f\n", model_total_ms);
    printf("model_ms_per_frame=%.6f\n", model_ms_per_frame);
    printf("model_realtime_budget_ms=4.000000\n");
    printf("model_realtime_ratio=%.6f\n", model_ms_per_frame / 4.0);
    printf("dsp_hops=%d\n", dsp_hops);
    printf("dsp_total_ms=%.3f\n", dsp_total_ms);
    printf("dsp_ms_per_hop=%.6f\n", dsp_ms_per_hop);
    printf("dsp_realtime_budget_ms=4.000000\n");
    printf("dsp_realtime_ratio=%.6f\n", dsp_ms_per_hop / 4.0);
    printf("sizeof_TinyTcnState=%zu\n", sizeof(TinyTcnState));
    printf("sizeof_TinyRealtimeDspState=%zu\n", sizeof(TinyRealtimeDspState));
    printf("sizeof_model_frame_input=%zu\n", sizeof(float) * (size_t)TINY_TCN_FEATURE_DIM);
    printf("sizeof_model_frame_output_q15=%zu\n", sizeof(int16_t) * (size_t)TINY_TCN_BANDS);
    return 0;
}
