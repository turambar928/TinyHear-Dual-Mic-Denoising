# 汇报 PPT 提纲

## 1. 标题页

题目：微听双麦降噪：面向助听器的端侧 AI 上行降噪原型

建议副标题：

- TinyHear Dual-Mic Denoising
- 100-150 KB 小模型、INT8/Q15、实时双麦增强

## 2. 背景与问题

要点：

- 助听器/耳戴设备上行通话容易受环境噪声影响。
- 设备端资源受限：功耗、SRAM、实时性、模型大小。
- 双麦可以提供空间信息，但需要端侧可运行的算法链路。

讲述重点：

- 不是只做一个离线降噪模型，而是要面向端侧实时部署。

## 3. 项目目标

要点：

- 双麦输入，单通道增强输出。
- 16 kHz，256 FFT，64 hop，4 ms 步进。
- 模型因果，不使用未来帧。
- INT8 权重 100-150 KB。
- 支持 C reference 和实时链路验证。

## 4. 总体方案

建议画图：

```text
Dual Mic PCM
  -> STFT
  -> 32-band features
  -> TinyCausalTCN
  -> 32-band mask
  -> mask-to-bin
  -> IRFFT/OLA
  -> enhanced speech
```

讲述重点：

- 模型只预测 mask，DSP 负责频域重构。
- 这种结构比直接生成 waveform 更适合端侧。

## 5. 特征设计

要点：

- mic0 log band power：32 维。
- mic1 log band power：32 维。
- mic0/mic1 log ratio：32 维。
- 总输入 96 维。

讲述重点：

- 双麦能量差和比例特征提供空间/噪声线索。

## 6. 模型结构

要点：

- Causal depthwise-separable TCN。
- 8 个 block。
- dilation = 1/2/4/8 repeat。
- 参数量 121,104。
- INT8 权重约 121 KB。

建议图：

```text
1x1 Conv
  -> DS-TCN x8
  -> 1x1 Conv
  -> hard sigmoid mask
```

## 7. 数据集与训练

要点：

- Clean speech：CMU ARCTIC。
- Noise：DEMAND 多通道环境噪声。
- 训练：动态 clean/noise 混音。
- 固定验证：160 条 materialized mixtures。
- 训练 40 epoch，后 20 epoch 低学习率续训。

讲述重点：

- ARCTIC + DEMAND 比 YESNO baseline 更接近真实语音和环境噪声。

## 8. 实验结果

建议表格：

| 指标 | 数值 |
| --- | ---: |
| INT8 权重 | 约 121 KB |
| Offline SI-SDR improvement | 4.749 dB |
| Realtime SI-SDR improvement | 4.740 dB |
| Realtime delay | 12 ms |
| Realtime vs offline delta | -0.009 dB |

讲述重点：

- 实时链路和离线路径几乎一致。
- 模型大小满足目标。

## 9. 量化与 C Reference

要点：

- int8 activation。
- int8 weight。
- int32 accumulate。
- Q15 mask output。
- C streaming 和 batch Q15 输出完全一致。

建议表格：

| C reference | 数值 |
| --- | ---: |
| Q15 model mean abs diff | 0.01477 |
| C realtime DSP mean abs diff | 0.00111 |
| TinyRealtimeDspState | 17.5 KB |

## 10. 实时链路

要点：

- Python realtime DSP 已实现。
- C realtime DSP reference 已实现。
- C FFT backend 已抽象。
- 当前 PC backend 是 naive DFT/IDFT。
- 后续可替换 CMSIS-DSP。

讲述重点：

- 项目不是停留在 PyTorch；已经推进到 C reference。

## 11. Benchmark

建议表格：

| 项目 | PC reference |
| --- | ---: |
| Q15 model | 0.267 ms/frame |
| Full C realtime reference | 2.530 ms/hop |
| Hop budget | 4.000 ms |
| TinyTcnState | 13.4 KB |
| TinyRealtimeDspState | 17.5 KB |

说明：

- 这是 x86/Linux PC reference，不是 MCU 上板性能。
- 真正上板需要 CMSIS-DSP/CMSIS-NN 后实测。

## 12. 工程完成度

可以用 checklist：

- 数据准备：完成。
- 训练与续训：完成。
- 固定验证集评估：完成。
- INT8 导出与校准：完成。
- Python realtime：完成。
- C Q15 streaming：完成。
- C realtime DSP reference：完成。
- benchmark：完成。
- GitHub CI：完成。

## 13. 当前限制

要点：

- 训练数据还不是真实设备双麦。
- C FFT backend 仍是 naive reference。
- C conv kernel 还没有替换 CMSIS-NN/U55。
- 量化还可以继续优化。
- 听感评估还可以补充。

## 14. 后续计划

优先级：

1. CMSIS-DSP FFT backend。
2. CMSIS-NN / Ethos-U55 conv kernel。
3. M55/U55 cycle 和 SRAM 实测。
4. 真实双麦数据微调。
5. 听感评估和展示样例。

## 15. 总结

一句话总结：

本项目完成了一个 121 KB INT8 双麦端侧降噪原型，并打通了训练、量化、Python realtime、C Q15 streaming、C realtime DSP reference 和 benchmark，为后续上板优化提供了完整基线。
