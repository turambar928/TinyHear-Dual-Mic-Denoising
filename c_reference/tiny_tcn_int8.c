#include "tiny_tcn_int8.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../runs/yesno_public/int8/model_int8.h"
#include "generated/model_config.h"
#include "generated/model_requant.h"

static void quantize_i8(const float *x, int n, float scale, int8_t *q) {
    for (int i = 0; i < n; ++i) {
        int v = (int)(x[i] / scale + (x[i] >= 0.0f ? 0.5f : -0.5f));
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
}

static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

static void add_inplace(float *x, const float *residual, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] += residual[i];
    }
}

static void hard_sigmoid_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) {
        float y = x[i] * 0.2f + 0.5f;
        if (y < 0.0f) y = 0.0f;
        if (y > 1.0f) y = 1.0f;
        x[i] = y;
    }
}

static int32_t rounding_divide_by_power_of_two(int64_t x, int exponent) {
    if (exponent <= 0) return (int32_t)x;
    int64_t mask = ((int64_t)1 << exponent) - 1;
    int64_t remainder = x & mask;
    int64_t threshold = ((int64_t)1 << (exponent - 1));
    if (x < 0) threshold -= 1;
    return (int32_t)((x >> exponent) + (remainder > threshold ? 1 : 0));
}

static int32_t multiply_by_quantized_multiplier(int32_t x, int32_t multiplier, int shift) {
    int64_t prod = (int64_t)x * (int64_t)multiplier;
    if (shift >= 0) {
        prod <<= shift;
        return rounding_divide_by_power_of_two(prod, 31);
    }
    return rounding_divide_by_power_of_two(prod, 31 - shift);
}

static int8_t clamp_i8(int32_t x) {
    if (x > 127) return 127;
    if (x < -127) return -127;
    return (int8_t)x;
}

static int16_t clamp_q15(int32_t x) {
    if (x > 32767) return 32767;
    if (x < 0) return 0;
    return (int16_t)x;
}

static void conv1d_i8_requant(
    const int8_t *x,
    int frames,
    const int8_t *w,
    const int32_t *bias,
    int out_ch,
    int in_per_group,
    int kernel,
    int groups,
    int dilation,
    int left_pad,
    int layer_idx,
    int relu,
    int8_t *y
) {
    int out_per_group = out_ch / groups;
    for (int oc = 0; oc < out_ch; ++oc) {
        int group = oc / out_per_group;
        int ic_start = group * in_per_group;
        for (int t = 0; t < frames; ++t) {
            int32_t acc = bias[oc];
            for (int icg = 0; icg < in_per_group; ++icg) {
                int ic = ic_start + icg;
                for (int k = 0; k < kernel; ++k) {
                    int src_t = t + k * dilation - left_pad;
                    if (src_t < 0 || src_t >= frames) continue;
                    int x_idx = ic * frames + src_t;
                    int w_idx = (oc * in_per_group + icg) * kernel + k;
                    acc += (int32_t)x[x_idx] * (int32_t)w[w_idx];
                }
            }
            int32_t q = multiply_by_quantized_multiplier(acc, kRequantMultipliers[layer_idx], kRequantShifts[layer_idx]);
            if (relu && q < 0) q = 0;
            y[oc * frames + t] = clamp_i8(q);
        }
    }
}

static void residual_add_requant(
    int8_t *x,
    const int8_t *residual,
    int n,
    int block
) {
    for (int i = 0; i < n; ++i) {
        int32_t res_q = multiply_by_quantized_multiplier(
            residual[i],
            kResidualMultipliers[block],
            kResidualShifts[block]
        );
        int32_t sum = (int32_t)x[i] + res_q;
        if (sum < 0) sum = 0;
        x[i] = clamp_i8(sum);
    }
}

static void conv1d_i8_dequant(
    const int8_t *x,
    int in_ch,
    int frames,
    const int8_t *w,
    const int32_t *bias,
    int out_ch,
    int in_per_group,
    int kernel,
    int groups,
    int dilation,
    int left_pad,
    float x_scale,
    float w_scale,
    float *y
) {
    (void)in_ch;
    int out_per_group = out_ch / groups;
    for (int oc = 0; oc < out_ch; ++oc) {
        int group = oc / out_per_group;
        int ic_start = group * in_per_group;
        for (int t = 0; t < frames; ++t) {
            int32_t acc = 0;
            for (int icg = 0; icg < in_per_group; ++icg) {
                int ic = ic_start + icg;
                for (int k = 0; k < kernel; ++k) {
                    int src_t = t + k * dilation - left_pad;
                    if (src_t < 0 || src_t >= frames) continue;
                    int x_idx = ic * frames + src_t;
                    int w_idx = (oc * in_per_group + icg) * kernel + k;
                    acc += (int32_t)x[x_idx] * (int32_t)w[w_idx];
                }
            }
            float bias_float = (float)bias[oc] * w_scale;
            y[oc * frames + t] = (float)acc * x_scale * w_scale + bias_float;
        }
    }
}

static void run_conv(
    const float *x_float,
    int in_ch,
    int frames,
    const int8_t *w,
    const int32_t *bias,
    int out_ch,
    int in_per_group,
    int kernel,
    int groups,
    int dilation,
    int left_pad,
    int layer_idx,
    float *y_float
) {
    int8_t *x_q = (int8_t *)malloc((size_t)in_ch * (size_t)frames);
    quantize_i8(x_float, in_ch * frames, kActivationScales[layer_idx], x_q);
    conv1d_i8_dequant(
        x_q,
        in_ch,
        frames,
        w,
        bias,
        out_ch,
        in_per_group,
        kernel,
        groups,
        dilation,
        left_pad,
        kActivationScales[layer_idx],
        kWeightScales[layer_idx],
        y_float
    );
    free(x_q);
}

void tiny_tcn_forward(const float *input, int frames, float *output) {
    const int channels = TINY_TCN_CHANNELS;
    float *a = (float *)calloc((size_t)channels * (size_t)frames, sizeof(float));
    float *b = (float *)calloc((size_t)channels * (size_t)frames, sizeof(float));
    float *residual = (float *)calloc((size_t)channels * (size_t)frames, sizeof(float));
    float *head = (float *)calloc((size_t)TINY_TCN_BANDS * (size_t)frames, sizeof(float));

    run_conv(input, TINY_TCN_FEATURE_DIM, frames, stem_0_weight, stem_0_bias,
             channels, TINY_TCN_FEATURE_DIM, 1, 1, 1, 0, 0, a);
    relu_inplace(a, channels * frames);

    const int8_t *dw_w[8] = {
        tcn_0_depthwise_weight, tcn_1_depthwise_weight, tcn_2_depthwise_weight, tcn_3_depthwise_weight,
        tcn_4_depthwise_weight, tcn_5_depthwise_weight, tcn_6_depthwise_weight, tcn_7_depthwise_weight
    };
    const int32_t *dw_b[8] = {
        tcn_0_depthwise_bias, tcn_1_depthwise_bias, tcn_2_depthwise_bias, tcn_3_depthwise_bias,
        tcn_4_depthwise_bias, tcn_5_depthwise_bias, tcn_6_depthwise_bias, tcn_7_depthwise_bias
    };
    const int8_t *pw_w[8] = {
        tcn_0_pointwise_weight, tcn_1_pointwise_weight, tcn_2_pointwise_weight, tcn_3_pointwise_weight,
        tcn_4_pointwise_weight, tcn_5_pointwise_weight, tcn_6_pointwise_weight, tcn_7_pointwise_weight
    };
    const int32_t *pw_b[8] = {
        tcn_0_pointwise_bias, tcn_1_pointwise_bias, tcn_2_pointwise_bias, tcn_3_pointwise_bias,
        tcn_4_pointwise_bias, tcn_5_pointwise_bias, tcn_6_pointwise_bias, tcn_7_pointwise_bias
    };
    const int dilations[8] = {1, 2, 4, 8, 1, 2, 4, 8};

    for (int block = 0; block < 8; ++block) {
        memcpy(residual, a, (size_t)channels * (size_t)frames * sizeof(float));
        int depth_layer = 1 + block * 2;
        int point_layer = depth_layer + 1;
        int left_pad = (TINY_TCN_KERNEL - 1) * dilations[block];
        run_conv(a, channels, frames, dw_w[block], dw_b[block],
                 channels, 1, TINY_TCN_KERNEL, channels, dilations[block], left_pad, depth_layer, b);
        relu_inplace(b, channels * frames);
        run_conv(b, channels, frames, pw_w[block], pw_b[block],
                 channels, channels, 1, 1, 1, 0, point_layer, a);
        relu_inplace(a, channels * frames);
        add_inplace(a, residual, channels * frames);
    }

    run_conv(a, channels, frames, head_weight, head_bias,
             TINY_TCN_BANDS, channels, 1, 1, 1, 0, 17, head);
    hard_sigmoid_inplace(head, TINY_TCN_BANDS * frames);
    memcpy(output, head, (size_t)TINY_TCN_BANDS * (size_t)frames * sizeof(float));

    free(a);
    free(b);
    free(residual);
    free(head);
}

void tiny_tcn_forward_q15(const float *input, int frames, int16_t *output_q15) {
    const int channels = TINY_TCN_CHANNELS;
    int8_t *input_q = (int8_t *)malloc((size_t)TINY_TCN_FEATURE_DIM * (size_t)frames);
    int8_t *a = (int8_t *)calloc((size_t)channels * (size_t)frames, sizeof(int8_t));
    int8_t *b = (int8_t *)calloc((size_t)channels * (size_t)frames, sizeof(int8_t));
    int8_t *residual = (int8_t *)calloc((size_t)channels * (size_t)frames, sizeof(int8_t));
    int8_t *head_q = (int8_t *)calloc((size_t)TINY_TCN_BANDS * (size_t)frames, sizeof(int8_t));

    quantize_i8(input, TINY_TCN_FEATURE_DIM * frames, kActivationScales[0], input_q);
    conv1d_i8_requant(input_q, frames, stem_0_weight, kBias_stem_0,
                      channels, TINY_TCN_FEATURE_DIM, 1, 1, 1, 0, 0, 1, a);

    const int8_t *dw_w[8] = {
        tcn_0_depthwise_weight, tcn_1_depthwise_weight, tcn_2_depthwise_weight, tcn_3_depthwise_weight,
        tcn_4_depthwise_weight, tcn_5_depthwise_weight, tcn_6_depthwise_weight, tcn_7_depthwise_weight
    };
    const int32_t *dw_b[8] = {
        kBias_tcn_0_depthwise, kBias_tcn_1_depthwise, kBias_tcn_2_depthwise, kBias_tcn_3_depthwise,
        kBias_tcn_4_depthwise, kBias_tcn_5_depthwise, kBias_tcn_6_depthwise, kBias_tcn_7_depthwise
    };
    const int8_t *pw_w[8] = {
        tcn_0_pointwise_weight, tcn_1_pointwise_weight, tcn_2_pointwise_weight, tcn_3_pointwise_weight,
        tcn_4_pointwise_weight, tcn_5_pointwise_weight, tcn_6_pointwise_weight, tcn_7_pointwise_weight
    };
    const int32_t *pw_b[8] = {
        kBias_tcn_0_pointwise, kBias_tcn_1_pointwise, kBias_tcn_2_pointwise, kBias_tcn_3_pointwise,
        kBias_tcn_4_pointwise, kBias_tcn_5_pointwise, kBias_tcn_6_pointwise, kBias_tcn_7_pointwise
    };
    const int dilations[8] = {1, 2, 4, 8, 1, 2, 4, 8};

    for (int block = 0; block < 8; ++block) {
        memcpy(residual, a, (size_t)channels * (size_t)frames);
        int depth_layer = 1 + block * 2;
        int point_layer = depth_layer + 1;
        int left_pad = (TINY_TCN_KERNEL - 1) * dilations[block];
        conv1d_i8_requant(a, frames, dw_w[block], dw_b[block],
                          channels, 1, TINY_TCN_KERNEL, channels, dilations[block], left_pad, depth_layer, 1, b);
        conv1d_i8_requant(b, frames, pw_w[block], pw_b[block],
                          channels, channels, 1, 1, 1, 0, point_layer, 1, a);
        residual_add_requant(a, residual, channels * frames, block);
    }

    conv1d_i8_requant(a, frames, head_weight, kBias_head,
                      TINY_TCN_BANDS, channels, 1, 1, 1, 0, 17, 0, head_q);
    for (int i = 0; i < TINY_TCN_BANDS * frames; ++i) {
        int32_t delta = multiply_by_quantized_multiplier(head_q[i], kHardSigmoidMultiplier, kHardSigmoidShift);
        output_q15[i] = clamp_q15(delta + 16384);
    }

    free(input_q);
    free(a);
    free(b);
    free(residual);
    free(head_q);
}
