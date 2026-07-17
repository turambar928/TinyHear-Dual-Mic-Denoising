# 双麦上行降噪实现方案

## 1. 项目目标

先实现一个可训练、可量化、可离线验证的双麦上行降噪原型。当前阶段不做完整助听器听力补偿，只处理“两个麦克风输入，输出更干净的上行语音”。

约束：

- 模型大小：INT8 权重 100-150 KB。
- 推理：因果，不使用未来帧。
- 端侧：STFT/ISTFT、特征和 mask 后处理在 M55/DSP；网络在 U55 或 M55 SIMD 上跑。
- 运算：模型主体按 int8 权重、int8 激活、int32 累加设计。

## 2. 算法流水线

```text
mic0/mic1 waveform
  -> causal STFT
  -> 32-band feature extraction
  -> TinyCausalTCN predicts 32-band soft mask
  -> interpolate mask to FFT bins
  -> apply mask to mic0 complex spectrum
  -> ISTFT/OLA
```

输入特征每帧 96 维：

- `log_band_power(mic0)`：32 维。
- `log_band_power(mic1)`：32 维。
- `log((power_mic0 + eps) / (power_mic1 + eps))`：32 维。

目标标签为频带理想幅度掩蔽：

```text
mask = sqrt(clean_power / (noisy_ref_power + eps))
mask = clamp(mask, min_gain, 1.0)
```

默认 `min_gain=0.08`，约等于最大抑制 22 dB。助听器场景里不建议把环境音完全切掉。

## 3. 模型架构

`TinyCausalTCN(feature_dim=96, bands=32, channels=112, blocks=8, kernel_size=5)`

结构：

```text
1x1 Conv + ReLU
8 x [
  causal depthwise Conv1d(k=5, dilation=1/2/4/8 repeat)
  ReLU
  1x1 Conv
  ReLU
  residual
]
1x1 Conv -> hard sigmoid -> mask
```

参数估算：

- stem：96 x 112 + 112 = 10,864。
- 每个 DS-TCN block：depthwise 约 672，pointwise 约 12,656，合计约 13,328。
- 8 blocks：约 106,624。
- head：112 x 32 + 32 = 3,616。
- 总计：约 121K 参数，INT8 权重约 121 KB。

实际脚本会打印精确参数量。

## 4. 双麦数据合成

公开数据集优先方案：

- 语音：DNS Challenge clean speech 或 LibriSpeech/Mini LibriSpeech。
- 噪声：DNS Challenge noise 或 MUSAN noise/music。
- RIR：DNS Challenge RIR，或者先用本项目的简单双麦延迟/衰减模拟。

当前代码提供两种模式：

- 预生成合成数据：`scripts/make_synth_dataset.py`，便于快速跑通。
- 训练时动态混音：`scripts/train.py --on-the-fly`，用已有 clean/noise wav 随机合成。

双麦合成假设目标语音在正前方，两个麦克风语音几乎同相；噪声随机方向，按麦间距和入射角产生 TDOA：

```text
delay_samples = mic_distance * sin(theta) / speed_of_sound * sample_rate
```

这个模拟不等于真实耳廓/头影/HRTF，但足以让模型先学会利用能量差和空间差异。

## 5. 量化与整型部署

当前实现提供 `scripts/export_int8.py`：

- 将每个卷积权重量化为 int8。
- 导出每层权重 scale、bias scale 和 JSON 元数据。
- 生成 `model_int8.h`，端侧 C 工程可直接包含。

端侧推理建议：

- 激活：int8，按层或按张量 scale。
- 权重：int8 per-tensor；后续可升级 per-channel。
- 累加：int32。
- ReLU：int8 clamp。
- 输出 hard sigmoid：用查表或分段线性整数近似。

如果目标平台使用 TensorFlow Lite Micro + Ethos-U55，下一步应把 PyTorch 权重迁移到 TFLite QAT 或直接重写为 Keras Conv1D/Conv2D equivalent，再用 Vela 编译。

## 6. 验证指标

第一阶段：

- 参数量和 INT8 文件大小。
- 验证集 mask MSE。
- SI-SDR improvement。
- 离线听感检查。

第二阶段：

- STOI/PESQ 或 DNSMOS。
- M55/U55 cycle 和 SRAM 峰值。
- 真实双麦录音上的泛化。

## 7. 当前已实现内容

- `scripts/prepare_wav_dataset.py`：将下载好的公开 wav 抽样整理成训练目录。
- `WavPairDataset --on-the-fly`：支持 clean/noise 动态混音，且可选 `rir` 目录。
- `scripts/evaluate.py`：输出 noisy/enhanced SI-SDR 和 mask MSE。
- `scripts/export_int8.py`：导出 int8 权重、int32 bias、C header 和 JSON 元数据。
- `scripts/verify_int8_reference.py`：用 int8 activation、int8 weight、int32 accumulate 验证量化权重路径。

端侧对接细节见 `docs/edge_deployment.md`。
