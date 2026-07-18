# 微听双麦降噪项目报告

## 1. 项目背景

助听器、耳戴设备和无线耳机在上行通话时需要从复杂环境中提取更清晰的语音。传统降噪方法通常依赖固定规则或较大的神经网络，而端侧设备受限于功耗、内存、实时性和模型大小，不能直接部署大型语音增强模型。

本项目实现一个面向助听器/耳戴设备的双麦端侧 AI 上行降噪原型。系统输入双通道麦克风音频，输出增强后的参考麦克风语音。目标是验证一个 100-150 KB 级别的小模型能否完成实时双麦降噪，并打通训练、量化、C reference 和实时链路。

## 2. 项目目标

核心目标：

- 双麦输入，单通道增强输出。
- 16 kHz 采样率，256 点 FFT，64 samples hop，4 ms 步进。
- 模型严格因果，不使用未来帧。
- INT8 权重大小控制在 100-150 KB。
- 支持 C 端 Q15/INT8 reference。
- 支持完整实时链路验证。

非目标：

- 当前阶段不做完整听力补偿。
- 当前阶段不直接声称 M55/U55 上板性能。
- 当前训练数据仍以公开数据集模拟为主，不等同真实助听器双麦录音。

## 3. 技术方案

系统流水线：

```text
mic0/mic1 waveform
  -> causal STFT
  -> 32-band feature extraction
  -> TinyCausalTCN predicts 32-band mask
  -> mask-to-bin interpolation
  -> apply mask to mic0 complex spectrum
  -> IRFFT + overlap-add
  -> enhanced waveform
```

每帧输入特征为 96 维：

- mic0 32-band log power。
- mic1 32-band log power。
- mic0/mic1 32-band log power ratio。

模型输出 32 维频带 mask，映射回 129 个 FFT bin 后乘到参考麦克风复数频谱。

## 4. 模型结构

模型为 causal depthwise-separable TCN：

```text
1x1 Conv + ReLU
8 x [
  causal depthwise Conv1d, kernel=5, dilation=1/2/4/8 repeat
  ReLU
  1x1 pointwise Conv
  ReLU
  residual
]
1x1 Conv
hard sigmoid
```

关键规模：

- 参数量：121,104。
- INT8 权重：约 121 KB。
- 输出：32-band soft mask。
- 整型路径：int8 activation、int8 weight、int32 accumulate、Q15 mask output。

这个结构的主要优势是参数量小、因果卷积易做流式状态缓存、depthwise + pointwise 分解适合端侧优化。

## 5. 数据与训练

当前推荐基线使用：

- Clean speech：CMU ARCTIC，`bdl` 和 `slt` 两个 speaker。
- Noise：DEMAND 16 kHz 多通道环境噪声。
- 噪声环境：`DKITCHEN`、`DLIVING`、`OHALLWAY`、`NFIELD`、`OOFFICE`、`PSTATION`。

训练方式：

- clean/noise 动态混音。
- 训练片段长度 2 秒。
- 总训练 40 epoch。
- 后 20 epoch 使用 `lr=3e-4` 从 best checkpoint 续训。

固定验证集：

- 160 条 ARCTIC + DEMAND materialized validation mixtures。

## 6. 实验结果

离线/实时核心指标：

| 指标 | 数值 |
| --- | ---: |
| 参数量 | 121,104 |
| INT8 权重 | 约 121 KB |
| ARCTIC + DEMAND offline SI-SDR improvement | 4.749 dB |
| Python realtime SI-SDR improvement | 4.740 dB |
| Realtime vs offline SI-SDR delta | -0.009 dB |
| 实时估计延迟 | 192 samples / 12 ms |
| TinyTcnState | 13,444 bytes |
| TinyRealtimeDspState | 17,540 bytes |

量化与 C reference：

| 项目 | 数值 |
| --- | ---: |
| Python fixed-scale max abs diff | 0.05847 |
| Python fixed-scale mean abs diff | 0.01182 |
| C Q15 model max abs diff | 0.12634 |
| C Q15 model mean abs diff | 0.01477 |
| C realtime DSP max abs diff | 0.38641 |
| C realtime DSP mean abs diff | 0.00111 |

PC reference benchmark：

| 项目 | 数值 |
| --- | ---: |
| Q15 model | 0.267 ms/frame |
| Full C realtime reference | 2.530 ms/hop |
| Realtime budget | 4.000 ms/hop |

说明：benchmark 是 x86/Linux 开发服务器上的 PC reference 数据，不能直接代表 M55/U55 上板性能。

## 7. 工程实现完成度

已完成模块：

- 数据准备脚本。
- 动态混音训练集。
- TinyCausalTCN 模型。
- 训练、续训、评估脚本。
- INT8 权重导出。
- activation calibration。
- Python fixed-scale reference。
- C Q15 模型 reference。
- C 单帧 streaming inference。
- Python 逐帧模型验证。
- Python 完整 realtime DSP。
- C 完整 realtime DSP reference。
- C benchmark。
- GitHub CI。
- 听感评估样例生成脚本和 5 组代表性样例。

## 8. 当前限制

主要限制：

- 训练数据仍是公开 clean speech + DEMAND 噪声模拟，不是真实助听器硬件双麦数据。
- C realtime DSP 当前默认 FFT backend 是 naive DFT/IDFT，只用于数值 reference。
- C 卷积 kernel 仍是 reference loop，没有替换为 CMSIS-NN 或 U55 加速 kernel。
- 量化使用 per-tensor weight scale，后续可升级为 per-channel 或 QAT。
- 已有 5 组听感样例用于展示，但还不是正式 MOS/ABX 测试。

## 9. 后续路线

建议后续优先级：

1. 用 CMSIS-DSP 替换 `fft_backend.c`。
2. 用 CMSIS-NN 或 Ethos-U55 替换 C conv loops。
3. 在 M55/U55 上实测 cycle、SRAM 峰值和 4 ms deadline。
4. 采集真实助听器/耳戴设备双麦数据做微调。
5. 加入听感评估样例和主观评分记录。
6. 尝试 per-channel quantization 或 QAT，降低 C Q15 max diff。

## 10. 结论

本项目已经打通从公开数据训练到端侧 C reference 的完整闭环。当前模型在 121 KB INT8 权重约束下实现了 4.74 dB 左右的实时 SI-SDR improvement，并提供 Python/C 两套 realtime reference。工程上已经具备继续替换 CMSIS-DSP/CMSIS-NN、上板测试和真实双麦数据微调的基础。
