# 微听双麦降噪项目介绍 PPT 提纲

对应文件：

```text
docs/presentation/tinyhear_project_intro.pptx
```

## 1. 标题

微听双麦降噪  
TinyHear Dual-Mic Denoising

面向助听器 / 耳戴设备上行链路的端侧 AI 降噪原型。

关键词：

- 双麦输入，单通道增强输出。
- 16 kHz，256 FFT，64 samples hop，4 ms 帧移。
- 当前主线模型 137,984 参数。
- 推荐版本：`teacher_deepfilter_conservative_on_wind_eval`。

## 2. 项目背景：为什么做双麦端侧降噪

应用场景：

- 助听器。
- 耳戴设备。
- 上行通话。

核心问题：

- 环境噪声。
- 风噪 / 气流呼呼声。
- 麦克风自噪声和摩擦噪声。

端侧约束：

- 低延迟。
- 小模型。
- 低功耗。
- 可定点化。

阶段目标：在 100K-150K 参数级别内，验证双麦空间信息 + 小模型是否能形成实时可展示的语音增强闭环。

## 3. 技术判断：近距双麦不能只看幅度

为什么不是单麦降噪：

- 双麦有空间信息，可以利用目标语音和噪声方向/相干性的差异。
- 单麦模型容易把风噪残留成低频呼呼声。

为什么重点用 IPD / coherence：

- 助听器/耳戴设备麦克风距离近，ILD 幅度差通常不明显。
- 相位差、相干性和局部空间统计更稳定。

当前路线：保留双麦空间前端，同时用轻量神经网络做细化增强。

## 4. 当前系统整体流程

推理流程：

```text
双麦输入
  -> STFT
  -> IPD / 相干性
  -> MWF 前端
  -> Tiny DeepFilter
  -> iSTFT
  -> 网页试听
```

评估流程：

- 固定风噪 eval 和 DEMAND eval。
- 输出 SI-SDR、RMS、failure analysis。
- 网页直接听 Noisy / Realtime / Clean。

## 5. 当前推荐模型结构

当前推荐模型：

```text
TinyDeepFilterTCN
```

关键配置：

| 模块 | 设计 |
| --- | --- |
| 前端 | coherence-weighted MWF spatial frontend |
| 主干网络 | TinyDeepFilterTCN，channels=96，blocks=8 |
| 输出 | band gain + low-frequency complex deep-filter coef |
| 时频配置 | 16 kHz，256 FFT，64 samples hop |
| 模型规模 | 137,984 参数 |

## 6. 数据集建设：从通用噪声到风噪

| 阶段 | Clean speech | Noise | 作用 |
| --- | --- | --- | --- |
| Baseline | 合成语音 | 合成噪声 | 快速打通训练/推理 |
| 小规模验证 | YESNO | DEMAND | sanity check |
| 主基线 | CMU ARCTIC | DEMAND | 环境噪声基线 |
| 当前主线 | CMU ARCTIC | DEMAND + Zenodo Wind | 风噪重点优化 |

训练目录：

```text
data/arctic_wind_demand_flat/
```

固定评估集：

```text
data/arctic_wind_eval/val
data/arctic_demand_eval/val
```

## 7. 版本迭代总览

主要版本路线：

```text
Baseline band-mask TCN
  -> Spatial features
  -> Tiny DeepFilter
  -> Postfilter / gate / GRU / complex mask
  -> Wind data fine-tune
  -> Conservative MWF teacher
```

结论：

- 单靠后处理或换一个小模型不能根治呼呼声。
- 更有效的方向是“更强双麦前端 + DeepFilter + MWF teacher 蒸馏”。

## 8. 为什么引入 MWF Teacher

Oracle local MWF 在 wind eval 上达到：

```text
+8.82 dB SI-SDR improvement
```

这说明强双麦前端 / MWF 是有效方向。

但 oracle MWF 训练时使用 clean/noise 信息，真实推理时没有 clean，所以不能直接部署。

当前采用保守蒸馏：

```text
target = 0.25 * oracle_mwf_teacher + 0.75 * clean_mic0
```

最终部署的仍然是 137,984 参数的小模型。

## 9. 训练目标与 Loss 设计

训练目标不是只优化 MSE 或单一 SI-SDR。

当前 loss 覆盖：

- waveform L1。
- SI-SDR loss。
- log STFT magnitude。
- residual noise loss。
- silence floor loss。
- coef energy regularization。

设计原则：

- 压低残留噪声。
- 保留 speech-active 区域的人声。
- 限制 deep-filter 复数系数过大，减少不稳定伪影。

## 10. 当前客观结果

推荐版本：

```text
teacher_deepfilter_conservative_on_wind_eval
```

checkpoint：

```text
runs/arctic_wind_teacher_deepfilter_mwf_conservative/best.pt
```

结果：

| 评估集 | Noisy | Enhanced | Improvement |
| --- | ---: | ---: | ---: |
| Zenodo wind eval | 5.39 dB | 10.19 dB | +4.80 dB |
| Original DEMAND eval | 4.33 dB | 10.96 dB | +6.63 dB |

## 11. 关键版本对比

| 版本 | Wind eval | DEMAND eval | 结论 |
| --- | ---: | ---: | --- |
| wind fine-tune | +4.35 dB | +7.06 dB | 通用噪声更稳，风噪仍有残留 |
| aggressive teacher | +3.74 dB | +5.21 dB | teacher 太强，小模型学不稳 |
| conservative teacher | +4.80 dB | +6.63 dB | 当前推荐，风噪更优 |

当前版本不是所有指标都最高，而是在风噪听感、人声保持和模型稳定性之间更均衡。

## 12. 失败样本分析

Wind eval failure mode：

| Failure mode | 数量 |
| --- | ---: |
| ok | 117 |
| low_freq_wind | 35 |
| regression | 7 |
| residual_noise | 1 |

诊断结论：

- 当前主要问题不是完全没有降噪能力。
- 主要瓶颈是低频风噪 / 气流残留。
- 后续应围绕 `low_freq_wind` 失败样本做定向优化。

## 13. 网页 Demo 展示方式

网页入口：

```text
http://127.0.0.1:38180/runs/audio_demo/index.html
```

建议展示：

1. 播放 Noisy，让听众感受原始噪声。
2. 播放 `teacher_deepfilter_conservative_on_wind_eval` 的 Realtime。
3. 播放 Clean，说明当前与理想 clean 的差距。

## 14. 已完成工程闭环

训练侧：

- 数据准备。
- 动态双麦混音。
- 多版本模型训练。
- teacher 蒸馏。

评估侧：

- SI-SDR metrics。
- 实时增强脚本。
- failure analysis。
- 固定样本对比。

展示/部署侧：

- 网页试听 demo。
- GitHub 文档。
- 早期 INT8 导出。
- C reference 链路。

## 15. 当前不足

当前仍存在：

- 相比 clean，部分样本仍有低频风噪和气流呼呼声。
- 公开数据集合成与真实助听器佩戴、真实风噪仍有 domain gap。
- SI-SDR 与主观听感不完全一致，需要固定听感评估集。
- 当前推荐 DeepFilter 版本还需要完成 INT8/C reference 对齐。

## 16. 下一步计划

下一步：

1. 低频风噪定向优化：固定 `low_freq_wind` 失败样本，增加低频 residual wind loss。
2. 语音保持约束：speech-active RMS / speech-band loss，避免继续压噪时人声变小。
3. 补充真实双麦数据：佩戴位置、走动风噪、摩擦噪声、麦克风自噪声和房间混响。
4. 端侧部署闭环：TinyDeepFilterTCN 的 INT8 导出、fixed-scale Python reference、C reference 对齐和 benchmark。
5. 展示完善：固定 demo 样本和版本对比，主观听感 + 客观指标同时汇报。

## 17. 阶段性结论

项目已经验证：

```text
强双麦前端 / MWF teacher 是有效方向。
```

当前推荐版本：

- Wind eval：`+4.80 dB SI-SDR improvement`。
- DEMAND eval：`+6.63 dB SI-SDR improvement`。

下一阶段重点不是盲目换模型，而是围绕低频风噪残留、真实双麦数据和端侧部署闭环继续收敛。

