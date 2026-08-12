# TinyHear 双麦端侧降噪项目阶段进展总结

更新时间：2026-08-12

## 1. 项目定位

本项目是一个面向助听器、耳戴设备和上行通话场景的双麦端侧 AI 降噪原型。目标是在 16 kHz 实时音频链路下，利用近距双麦的相位差、相干性和空间信息，输出更清晰的单通道语音。

核心约束：

- 输入：双通道麦克风音频。
- 输出：单通道增强语音。
- 采样率：16 kHz。
- STFT：256 点 FFT。
- hop：64 samples，即 4 ms 步进。
- 模型规模：约 100-150K 参数。
- 推理方向：端侧实时、因果处理，后续支持 INT8/C reference。

当前项目名称：

```text
TinyHear Dual-Mic Denoising
```

GitHub 仓库：

```text
https://github.com/turambar928/TinyHear-Dual-Mic-Denoising
```

## 2. 当前推荐版本

当前主线版本是：

```text
teacher_deepfilter_conservative_on_wind_eval
```

对应 checkpoint：

```text
runs/arctic_wind_teacher_deepfilter_mwf_conservative/best.pt
```

对应详细设计文档：

```text
docs/conservative_mwf_teacher_deepfilter.md
```

当前推荐网页试听入口：

```text
http://127.0.0.1:38180/runs/audio_demo/index.html
```

推荐在网页里优先听：

```text
teacher_deepfilter_conservative_on_wind_eval
```

然后对比：

```text
wind_finetune_on_wind_eval
teacher_deepfilter_aggressive_on_wind_eval
teacher_deepfilter_conservative_on_demand_eval
```

## 3. 当前技术路线

当前主线可以概括为：

```text
双麦输入
  -> STFT
  -> IPD / coherence 等空间特征
  -> coherence-weighted MWF spatial frontend
  -> Tiny DeepFilter TCN
  -> band gain + low-frequency deep filtering
  -> iSTFT / overlap-add
  -> enhanced speech
```

训练阶段额外加入：

```text
oracle local MWF teacher
```

它不是最终部署模型，而是训练时用 clean 信息构造出的强老师。真实推理时没有 clean，所以不会使用 teacher，只使用 Tiny DeepFilter 小模型。

当前采用的是保守 teacher 蒸馏：

```text
target = 0.25 * oracle_mwf_teacher + 0.75 * clean_mic0
```

这样做是因为之前实验发现，teacher 目标太强会让小模型过度滤波，导致人声变小、通用噪声场景退化。

## 4. 模型结构

当前推荐模型：

```text
TinyDeepFilterTCN
```

主要配置：

| 项目 | 数值 |
| --- | ---: |
| 参数量 | 137,984 |
| channels | 96 |
| blocks | 8 |
| kernel size | 5 |
| df bins | 64 |
| df order | 3 |
| spatial frontend | coherence_mwf |
| sample rate | 16 kHz |
| FFT | 256 |
| hop | 64 samples |

模型输出：

- `gain`：频带增益，用于整体谱幅度抑制。
- `coef`：低频 deep-filter 复数滤波系数，用于多帧/复数域细化，缓解单纯幅度 mask 的相位和残留噪声问题。

## 5. 数据进展

项目先后使用过的数据组合：

| 阶段 | clean speech | noise | 作用 |
| --- | --- | --- | --- |
| synthetic baseline | 合成语音 | 合成噪声 | 打通训练和推理流程 |
| YESNO + DEMAND | YESNO | DEMAND | 小规模公开数据 sanity check |
| ARCTIC + DEMAND | CMU ARCTIC | DEMAND | 主基线 |
| ARCTIC + DEMAND + wind | CMU ARCTIC | DEMAND + Zenodo Wind Noise | 当前风噪优化主线 |

当前主线训练目录：

```text
data/arctic_wind_demand_flat/
```

当前固定评估集：

```text
data/arctic_wind_eval/val
data/arctic_demand_eval/val
```

说明：

- 原始下载数据和整理后的大数据不提交到 GitHub。
- GitHub 中保留 checkpoint、metrics、少量网页试听样本和分析报告。

## 6. 版本演进

项目从最初到现在主要经历了以下阶段。

### 6.1 Tiny band-mask TCN

最初版本使用轻量 causal TCN 预测 32-band mask。

特点：

- 模型小。
- 易于导出 INT8。
- 易于做 C reference。
- 能实现基本降噪。

问题：

- 人声容易变小。
- 底噪明显。
- 只做幅度 mask，不能处理 noisy phase 和细粒度残留噪声。

### 6.2 空间特征增强

后续加入双麦空间特征：

- IPD。
- coherence。
- 双麦能量比。

原因是近距双麦下 ILD 差异很小，IPD 和相干性更有价值。

### 6.3 learned gate / bypass

尝试通过 gate 或 bypass 减少干净/高 SNR 场景下的过处理。

结论：

- 对部分样本有帮助。
- 但不能根治底噪和呼呼声。

### 6.4 Tiny DeepFilter

将后端从简单 band mask 升级到 Tiny DeepFilter。

核心变化：

- 不只输出 band gain。
- 额外输出 deep-filter 复数系数。
- 能做多帧/复数域滤波。

结果：

- SI-SDR 指标明显提升。
- 主观听感比早期 band-mask 版本更好。
- 但仍存在风噪/气流类残留声。

### 6.5 coherence/MWF spatial frontend

前端从简单 delay-sum 升级到 coherence-weighted MWF 风格空间前端。

作用：

- 利用双麦相干性抑制不稳定噪声。
- 给 DeepFilter 后端提供更好的参考通道。

结论：

- 指标和听感都有提升。
- 成为后续主线基础。

### 6.6 postfilter / dehiss / airflow filter

尝试过多种后处理：

- stable postfilter。
- dehiss postfilter。
- airflow residual filter。

结论：

- 可以局部降低高频 hiss 或部分残留。
- 但不能从根本上解决风噪呼呼声。
- 后处理更适合作为辅助，不适合作为主解。

### 6.7 TinyGRU / complex mask / student 分支

尝试过：

- RNNoise-style TinyGRU。
- Tiny complex-mask TCN。
- MWF covariance student。

结论：

- 一些版本指标不错，但主观听感不稳定。
- complex-mask 和 student 分支没有明显优于 DeepFilter 主线。
- 项目最终回到 Tiny DeepFilter + 强双麦前端方向。

### 6.8 风噪数据微调

引入 Zenodo Wind Noise Dataset 后，将真实风噪片段加入训练。

普通风噪微调版：

```text
runs/arctic_wind_demand_tiny_deepfilter_coherence_mwf/best.pt
```

效果：

| Eval set | SI-SDR improvement |
| --- | ---: |
| Zenodo wind eval | +4.35 dB |
| DEMAND eval | +7.06 dB |

结论：

- 风噪数据确实有效。
- 但听感上仍有低频呼呼声残留。

### 6.9 MWF teacher 蒸馏

用 oracle local MWF 做 teacher 上限验证：

```text
Zenodo wind eval:
Noisy SI-SDR:     5.39 dB
Enhanced SI-SDR: 14.21 dB
Improvement:     +8.82 dB
```

说明强双麦前端/MWF 方向有明显上限。

随后训练两个 teacher 版本：

| 版本 | teacher blend | wind eval | DEMAND eval | 结论 |
| --- | ---: | ---: | ---: | --- |
| aggressive teacher | 0.90 | +3.74 dB | +5.21 dB | 过强 teacher 目标导致退化 |
| conservative teacher | 0.25 | +4.80 dB | +6.63 dB | 当前推荐风噪版本 |

当前推荐使用 conservative teacher 版本。

## 7. 当前指标

当前推荐模型：

```text
runs/arctic_wind_teacher_deepfilter_mwf_conservative/best.pt
```

### 7.1 Zenodo wind eval

| 指标 | 数值 |
| --- | ---: |
| items | 160 |
| noisy SI-SDR | 5.39 dB |
| enhanced SI-SDR | 10.19 dB |
| SI-SDR improvement | +4.80 dB |
| output/input RMS ratio | 0.764 |

对比普通 wind fine-tune：

| 模型 | SI-SDR improvement |
| --- | ---: |
| wind fine-tune | +4.35 dB |
| conservative MWF teacher | +4.80 dB |

### 7.2 Original DEMAND eval

| 指标 | 数值 |
| --- | ---: |
| items | 160 |
| noisy SI-SDR | 4.33 dB |
| enhanced SI-SDR | 10.96 dB |
| SI-SDR improvement | +6.63 dB |
| output/input RMS ratio | 0.690 |

对比普通 wind fine-tune：

| 模型 | SI-SDR improvement |
| --- | ---: |
| wind fine-tune | +7.06 dB |
| conservative MWF teacher | +6.63 dB |

结论：

- 当前版本更偏风噪抑制。
- 在 DEMAND 通用噪声上有轻微回退。
- 这属于风噪目标和通用噪声目标之间的取舍。

## 8. 失败样本分析

为了避免只靠主观听感，新增了失败模式分析工具：

```text
scripts/analyze_failure_modes.py
```

输出结果：

```text
runs/arctic_wind_teacher_deepfilter_mwf_conservative/analysis_wind_zenodo/failure_analysis.md
runs/arctic_wind_teacher_deepfilter_mwf_conservative/analysis_demand_nomatch/failure_analysis.md
```

### 8.1 Wind eval 失败模式

| failure mode | 数量 |
| --- | ---: |
| ok | 117 |
| low_freq_wind | 35 |
| regression | 7 |
| residual_noise | 1 |

关键均值：

| 指标 | 数值 |
| --- | ---: |
| mean speech preservation | -0.96 dB |
| mean quiet noise reduction | 15.73 dB |
| mean quiet enhanced RMS | -44.32 dBFS |
| mean quiet low RMS | -40.45 dBFS |
| mean quiet high RMS | -71.64 dBFS |

结论：

- 人声整体没有被严重压小。
- 当前主要问题是部分样本的低频风噪/气流残留。
- 后续不应继续盲目加大整体降噪，而应针对低频风噪残留做定向处理。

### 8.2 DEMAND eval 失败模式

| failure mode | 数量 |
| --- | ---: |
| ok | 117 |
| low_freq_wind | 26 |
| regression | 13 |
| residual_noise | 1 |
| speech_too_small | 3 |

关键均值：

| 指标 | 数值 |
| --- | ---: |
| mean speech preservation | -1.62 dB |
| mean quiet noise reduction | 21.61 dB |

结论：

- DEMAND 场景下回退样本更多。
- 说明当前模型偏风噪优化后，通用环境噪声有一定代价。
- 但人声过小不是最主要问题。

## 9. 工程闭环

目前已经完成的工程能力：

- 数据准备脚本。
- 动态双麦混音数据集。
- TinyCausalTCN baseline。
- Tiny DeepFilter TCN。
- coherence/MWF 空间前端。
- MWF teacher 蒸馏训练。
- 多种 loss 组合。
- 离线评估。
- 实时增强脚本。
- 网页试听 demo。
- 失败样本分析脚本。
- 指标 JSON/CSV/Markdown 报告。
- INT8 导出和 C reference 的早期链路。
- GitHub push 和文档同步。

已提交的近期关键 commit：

```text
8c8ad43 Add failure mode analysis for wind teacher model
f51cf16 Document conservative MWF teacher design
6431b8b Add conservative MWF teacher wind fine tune
284d391 Add wind-noise dataset fine tune
```

## 10. 当前仍存在的问题

当前版本相较前几版已经有提升，但仍没有达到 clean 的听感。主要问题包括：

- 部分样本仍有低频风噪/气流呼呼声。
- 个别样本存在 SI-SDR regression。
- 风噪优化会牺牲一部分 DEMAND 通用噪声表现。
- 训练数据仍是公开数据集合成，和真实助听器佩戴位置、麦克风自噪声、真实风噪还有差距。
- SI-SDR 与主观听感不完全一致。
- 当前 Tiny DeepFilter 主线还没有完成最新版本的 INT8/C reference 导出闭环。

## 11. 下一步建议

下一步建议不要盲目换模型，而是围绕当前最有效主线继续做定向优化。

优先级 1：低频风噪定向优化

- 重点处理 `low_freq_wind` 类失败样本。
- 使用 `failure_analysis.md` 中列出的 worst residual 样本作为固定调参集。
- 增加低频风噪 residual loss，而不是提高全频抑制强度。

优先级 2：语音保持约束

- 增加 speech preservation loss。
- 约束 speech-active 区域 RMS 不要明显低于 clean。
- 避免后续压风噪时再次把人声压小。

优先级 3：更真实数据

- 补充真实双麦风噪数据。
- 重点采集佩戴位置、近距双麦、走动/吹风/摩擦噪声。
- 哪怕少量真实数据，也可以用于 fine-tune 或 failure set。

优先级 4：端侧部署闭环

- 对当前推荐 `TinyDeepFilterTCN` 版本做 INT8 导出。
- 对齐 Python fixed-scale reference。
- 扩展 C reference 到 DeepFilter 主线。
- benchmark 内存、状态、每帧耗时。

优先级 5：展示材料

- 网页 demo 保持当前推荐版本在最前。
- 报告中明确说明 teacher 只用于训练，不用于部署。
- 展示指标时同时展示 SI-SDR 和失败模式分析，避免只讲单一指标。

## 12. 阶段性结论

到目前为止，项目已经从最初的 Tiny band-mask TCN，推进到更适合当前问题的 **coherence/MWF spatial frontend + Tiny DeepFilter + conservative MWF teacher distillation** 路线。

当前版本的价值在于：

- 保持 137,984 参数的小模型规模。
- 使用双麦空间信息，而不是单麦降噪。
- 引入真实风噪数据。
- 用 oracle local MWF teacher 验证并蒸馏强双麦前端能力。
- 通过网页 demo 和 failure analysis 同时进行主观/客观评估。

当前最重要的技术判断是：

```text
项目现在的主要瓶颈不是有没有降噪能力，
而是低频风噪残留和真实双麦数据匹配度。
```

因此下一阶段应围绕低频风噪失败样本和真实双麦数据继续优化，而不是继续无目的地切换模型结构。
