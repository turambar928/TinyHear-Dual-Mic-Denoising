# M55/U55 端侧部署说明

## 1. 每帧接口

模型不直接吃 waveform，而是吃 DSP 侧提取的 32 频带特征。

输入 tensor：

```text
shape: [1, 96, T]
layout: channel-first
channels:
  0..31   log_band_power(mic0)
  32..63  log_band_power(mic1)
  64..95  log_power_ratio(mic0/mic1)
```

输出 tensor：

```text
shape: [1, 32, T]
value: 0.0..1.0 band mask
```

流式部署时建议 `T=1` 单帧推理，TCN 每层保存左侧历史缓存。离线 Python 为了训练效率用多帧 `[T]` 一起跑，但模型本身是因果卷积。

## 2. DSP 侧处理

M55/Helium：

1. 双麦 PCM 输入，16 kHz。
2. 256 点 Hann STFT，64 点 hop，4 ms 步进。
3. 计算两个麦克风 power spectrum。
4. 乘 32-band mel-like band matrix。
5. 取 log power 和 log ratio，量化为 int8 activation。
6. 调 U55/CMSIS-NN 跑 TinyCausalTCN。
7. 32-band mask 插值回 129 个 FFT bin。
8. mask 乘参考麦复数谱，ISTFT/OLA 输出。

## 3. 整型推理约定

当前 `scripts/export_int8.py` 导出：

- `model_int8.h`：C 数组，包含 int8 权重和 int32 bias 初值。
- `model_int8.json`：层名、shape、kernel、dilation、groups、weight scale。
- `*.npy`：Python reference 使用的同一份权重。

计算约定：

```text
input_int8 = round(input_float / input_scale)
weight_int8 = round(weight_float / weight_scale)
acc_int32 = conv(input_int8, weight_int8) + bias_int32
output_float ~= acc_int32 * input_scale * weight_scale
```

真正固化到端侧前，需要用校准集统计每层 activation scale。当前推荐基线使用 ARCTIC + DEMAND 固定验证集做 100th percentile 校准，以降低真实语音样本上的裁剪误差。

校准命令：

```bash
PYTHONPATH=src python scripts/calibrate_int8.py \
  --checkpoint runs/arctic_demand/best.pt \
  --export-dir runs/arctic_demand/int8 \
  --data data/arctic_demand_eval \
  --split val \
  --percentile 100

PYTHONPATH=src python scripts/verify_int8_reference.py \
  --checkpoint runs/arctic_demand/best.pt \
  --export-dir runs/arctic_demand/int8 \
  --fixed-scales
```

校准后 `model_int8.json` 会新增：

```json
{
  "activation_scales": {
    "stem.0": 0.123,
    "tcn.0.depthwise": 0.045
  },
  "calibration": {
    "items": 24,
    "percentile": 100.0
  }
}
```

端侧 C 工程应优先使用这些固定 scale，而不是 Python reference 里的动态 scale。

## 4. SRAM 估算

默认模型：

- 权重：约 121 KB int8。
- bias：约 5.8 KB int32。
- 单帧激活：`112` 通道为主，int8 下很小。
- TCN 历史缓存：每个 depthwise block 保存 `(kernel_size - 1) * dilation * channels`。

默认 8 个 block、dilation `[1,2,4,8,1,2,4,8]`、kernel 5、channels 112：

```text
history bytes ~= 112 * 4 * (1+2+4+8+1+2+4+8) = 13,440 bytes
```

实际还要加 STFT buffer、FFT twiddle、band power buffer 和输出 overlap-add buffer。

## 5. C 工程接入顺序

1. 先在 PC 上用 `verify_int8_reference.py` 确认导出权重误差。
2. 在 C 侧实现 Conv1D pointwise、depthwise causal conv、ReLU、residual、hard sigmoid。
3. 用同一组输入特征 dump 对齐 Python reference 的逐层输出。
4. 接入 CMSIS-NN 或 TFLite Micro kernel。
5. 用 Arm Vela 检查 Ethos-U55 算子落图情况。
6. 上板测 cycle、SRAM 峰值和每帧 4 ms deadline。

当前仓库已提供 PC 侧 C reference：

```bash
PYTHONPATH=src python scripts/dump_c_reference_assets.py \
  --checkpoint runs/arctic_demand/best.pt \
  --export-dir runs/arctic_demand/int8 \
  --input-wav data/arctic_demand_eval/val/mix_0000.wav \
  --out-dir c_reference/generated \
  --frames 16

make -C c_reference run
```

这个 reference 包含两条路径：

- `tiny_tcn_forward`：卷积本体是 int8 activation、int8 weight、int32 accumulate；scale/requantize 用 float，目的是和 Python fixed-scale reference 对齐。
- `tiny_tcn_forward_q15`：中间激活为 int8，bias 为 int32，requantize 使用 fixed-point multiplier/shift，最终 mask 输出为 Q15。
- `tiny_tcn_process_frame_q15`：端侧单帧流式接口，`TinyTcnState` 保存每个 TCN depthwise block 的历史激活。
- `tiny_realtime_process_hop`：完整 C realtime DSP reference，每次输入 2 x 64 samples，输出 64 samples 增强音频。

`scripts/dump_c_reference_assets.py` 会生成：

- `generated/model_config.h`：activation/weight scale。
- `generated/model_requant.h`：每层 int32 bias、requant multiplier/shift、residual multiplier/shift、hard-sigmoid Q15 参数。
- `generated/test_vectors.h`：输入特征和 PyTorch 期望输出。
- `generated/band_matrix.h`：129 x 32 band projection matrix。
- `generated/realtime_vectors.h`：双麦 hop 输入和 Python realtime 期望输出。

后续 CMSIS-NN/U55 版本可以用 `tiny_tcn_forward_q15` 的数值语义替换底层 conv kernel。
后续 CMSIS-DSP 版本可以用 `tiny_realtime_process_hop` 的数值语义替换朴素 DFT/IDFT。

## 6. 实时链路验证

Python 侧完整实时链路：

```bash
PYTHONPATH=src python scripts/enhance_realtime.py \
  --checkpoint runs/arctic_demand/best.pt \
  --input data/arctic_demand_eval/val/mix_0000.wav \
  --output runs/arctic_demand/realtime_eval/example.wav

PYTHONPATH=src python scripts/compare_realtime.py \
  --checkpoint runs/arctic_demand/best.pt \
  --data data/arctic_demand_eval \
  --split val
```

当前 ARCTIC + DEMAND 固定验证集结果：

- 实时链路：causal 256 点 STFT、64 hop、逐帧模型、IRFFT、overlap-add。
- 估计延迟：192 samples，16 kHz 下约 12 ms。
- 平均实时 SI-SDR improvement：4.740 dB。
- 相比离线 `center=True` 路径：平均 SI-SDR 低约 0.009 dB。

当前 C realtime DSP reference 使用朴素 DFT/IDFT，目的是便于 PC 侧数值对齐，不代表端侧性能。`make -C c_reference run` 同时验证：

- Q15 模型 streaming 和 batch 输出完全一致。
- 完整 C realtime DSP 相对 Python realtime float reference：max abs diff 0.386406660，mean abs diff 0.001111976。

## 7. Benchmark

PC 侧 benchmark：

```bash
make -C c_reference clean bench
```

当前开发服务器结果：

- Q15 streaming model：0.280816 ms/frame。
- Full realtime DSP reference：2.450210 ms/hop。
- `sizeof(TinyTcnState)`：13,444 bytes。
- `sizeof(TinyRealtimeDspState)`：17,540 bytes。

这些数值只用于 PC reference 追踪，不代表 M55/U55 上板性能。详细记录见 `docs/performance.md`。
