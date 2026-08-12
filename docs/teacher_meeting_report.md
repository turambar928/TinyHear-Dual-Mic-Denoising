# TinyHear 双麦端侧降噪项目汇报稿

面向老师汇报版本  
日期：2026-08-12

## 1. 开场介绍

老师好，我目前做的项目是 **TinyHear Dual-Mic Denoising**，中文可以叫“微听双麦降噪”。项目目标是做一个面向助听器、耳戴设备和上行通话场景的双麦端侧 AI 降噪原型。

这个项目不是单纯做一个离线语音增强模型，而是希望在端侧约束下完成一个比较完整的闭环：

- 双麦输入，单通道增强输出。
- 16 kHz 采样率。
- 256 点 FFT，64 samples hop，也就是 4 ms 一帧。
- 模型规模控制在 100-150K 参数左右。
- 后续能够支持 INT8/C reference 和实时部署。
- 通过网页 demo 直接听 noisy、enhanced、clean 的对比。

当前项目已经从最早的 band-mask TCN，迭代到现在的 **coherence/MWF 空间前端 + Tiny DeepFilter + 保守 MWF teacher 蒸馏** 版本。

## 2. 背景和问题

助听器或耳戴设备在上行通话时，常见问题是环境噪声、风噪、气流噪声会严重影响语音清晰度。和服务器端模型不同，端侧设备有几个限制：

- 算力有限。
- SRAM 和 Flash 有限。
- 功耗敏感。
- 必须实时处理，不能依赖大延迟。
- 模型不能太大。

双麦相比单麦有额外优势：可以利用两个麦克风之间的相位差、相干性和空间信息。但问题是助听器/耳戴设备上的麦克风距离很近，ILD 幅度差不明显，所以后续我重点使用 IPD 和 coherence 这类空间特征。

## 3. 项目目标

当前阶段目标可以概括为三点：

1. **算法可行性**

   验证 100-150K 级别的小模型，在双麦输入下是否能实现有效降噪。

2. **端侧可实现性**

   模型结构要适合实时因果推理，后续能导出 INT8，能接 C reference。

3. **听感可展示**

   不能只看 SI-SDR 指标，要能在网页 demo 里直接听 noisy、enhanced、clean 的差别，并分析失败样本。

## 4. 当前总体方案

当前主线流程是：

```text
双麦 waveform
  -> STFT
  -> 双麦空间特征提取
  -> coherence-weighted MWF spatial frontend
  -> Tiny DeepFilter TCN
  -> band gain + low-frequency deep filtering
  -> iSTFT / overlap-add
  -> enhanced speech
```

核心思路是：不要只依赖一个神经网络直接降噪，而是先用双麦空间前端给模型一个更好的参考通道，再用轻量神经网络做细化增强。

当前推荐模型是：

```text
runs/arctic_wind_teacher_deepfilter_mwf_conservative/best.pt
```

网页 demo 中对应的名字是：

```text
teacher_deepfilter_conservative_on_wind_eval
```

## 5. 模型结构

当前主线模型是 `TinyDeepFilterTCN`。

主要参数：

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

模型输出两类结果：

- `gain`：频带增益，用于整体谱幅度抑制。
- `coef`：deep-filter 复数系数，主要作用在低频部分，用多帧复数滤波减少单纯幅度 mask 带来的相位问题和残留噪声。

相比最早的 TCN band-mask 版本，现在的 DeepFilter 版本更适合处理“呼呼声”这类不稳定残留，因为它不是只改幅度，而是有一定复数域、多帧滤波能力。

## 6. 数据集进展

项目数据经历了几次升级：

| 阶段 | clean speech | noise | 用途 |
| --- | --- | --- | --- |
| synthetic baseline | 合成语音 | 合成噪声 | 快速打通流程 |
| YESNO + DEMAND | YESNO | DEMAND | 小规模公开数据验证 |
| ARCTIC + DEMAND | CMU ARCTIC | DEMAND | 主基线 |
| ARCTIC + DEMAND + wind | CMU ARCTIC | DEMAND + Zenodo Wind Noise | 当前风噪优化 |

当前主要训练集：

```text
data/arctic_wind_demand_flat/
```

固定评估集：

```text
data/arctic_wind_eval/val
data/arctic_demand_eval/val
```

其中 `arctic_wind_eval` 用来重点评估风噪，`arctic_demand_eval` 用来观察通用环境噪声上是否退化。

## 7. 版本迭代过程

项目目前经历了多轮优化。

### 7.1 初始 band-mask TCN

最早版本是一个 121K 左右的 causal TCN，输出 32-band mask。

优点：

- 模型小。
- 易于 INT8。
- 易于 C reference。

问题：

- 有基本降噪，但人声偏小。
- 底噪和呼呼声明显。
- 只做幅度 mask，对相位和细粒度残留处理不足。

### 7.2 空间特征和 coherence

之后加入 IPD、coherence 等双麦空间特征。原因是近距双麦中 ILD 不明显，相位差和相干性更重要。

这一步让模型能利用双麦信息，而不是退化成单麦降噪。

### 7.3 DeepFilter 版本

然后把后端从普通 mask TCN 升级到 Tiny DeepFilter。

DeepFilter 的主要变化是：

- 不只预测一个频带 mask。
- 额外预测复数滤波系数。
- 可以做低频多帧滤波。

这个版本客观指标提升明显，听感也比早期 band-mask 更好。

### 7.4 后处理尝试

我还尝试过：

- stable postfilter。
- dehiss。
- airflow filter。
- learned gate。
- high-SNR bypass。

结论是这些方法能局部改善，但不能根治底噪和呼呼声。所以后处理目前只作为辅助，不作为主线。

### 7.5 GRU、complex mask、student 分支

也尝试过：

- RNNoise-style TinyGRU。
- Tiny complex-mask TCN。
- MWF covariance student。

这些版本有些指标不错，但主观听感不稳定，尤其 complex-mask 和 student 分支没有明显超过 DeepFilter 主线。所以最后回到 DeepFilter + 强双麦前端方向。

### 7.6 风噪数据微调

后来加入 Zenodo Wind Noise Dataset，构造 ARCTIC + DEMAND + Wind 的训练集。

普通 wind fine-tune 结果：

| Eval set | SI-SDR improvement |
| --- | ---: |
| Zenodo wind eval | +4.35 dB |
| DEMAND eval | +7.06 dB |

这说明真实风噪数据是有效的，但听感上仍然有低频风噪残留。

## 8. MWF Teacher 是什么

现在推荐版本里有一个关键词叫 `teacher`。

这里的 teacher 不是最终部署模型，而是训练阶段使用的 oracle local MWF。它可以理解成一个“知道 clean 语音信息的强老师”。

训练时可以拿到：

- noisy 双麦。
- clean 双麦。

所以可以构造一个 oracle MWF teacher，输出一个更强的增强结果。但真实部署时没有 clean，所以这个 teacher 不能直接用在设备上。

它的作用是训练时指导小模型学习：

```text
哪些频段应该压噪
哪些地方应该保留人声
双麦相干性和空间信息应该怎样被利用
```

当前最终部署的仍然是 137,984 参数的 Tiny DeepFilter 小模型。

## 9. 为什么用 Conservative Teacher

我做过一个上限验证：oracle local MWF 在 wind eval 上效果明显更强。

```text
Oracle local MWF on Zenodo wind eval:
Noisy SI-SDR:     5.39 dB
Enhanced SI-SDR: 14.21 dB
Improvement:     +8.82 dB
```

这说明强双麦前端 / MWF 方向是有效的。

但直接让小模型强行模仿 teacher 效果不好。我训练过一个 aggressive teacher 版本：

```text
teacher_blend = 0.90
```

结果：

| 版本 | wind eval | DEMAND eval |
| --- | ---: | ---: |
| aggressive teacher | +3.74 dB | +5.21 dB |

反而比普通 wind fine-tune 更差。原因是 teacher 太强，小模型学不到完整行为，容易过度滤波，导致人声和通用噪声场景退化。

所以当前采用 conservative teacher：

```text
target = 0.25 * oracle_mwf_teacher + 0.75 * clean_mic0
```

对应结果：

| 版本 | wind eval | DEMAND eval |
| --- | ---: | ---: |
| wind fine-tune | +4.35 dB | +7.06 dB |
| conservative teacher | +4.80 dB | +6.63 dB |

也就是说，当前版本更偏向解决风噪，代价是 DEMAND 通用噪声上略有下降。

## 10. 当前最好结果

当前推荐 checkpoint：

```text
runs/arctic_wind_teacher_deepfilter_mwf_conservative/best.pt
```

### 10.1 Wind eval

| 指标 | 数值 |
| --- | ---: |
| items | 160 |
| noisy SI-SDR | 5.39 dB |
| enhanced SI-SDR | 10.19 dB |
| SI-SDR improvement | +4.80 dB |
| output/input RMS ratio | 0.764 |

### 10.2 DEMAND eval

| 指标 | 数值 |
| --- | ---: |
| items | 160 |
| noisy SI-SDR | 4.33 dB |
| enhanced SI-SDR | 10.96 dB |
| SI-SDR improvement | +6.63 dB |
| output/input RMS ratio | 0.690 |

从网页试听上看，当前版本相比之前版本风噪更小，人声也没有像早期版本那样明显偏小，但和 clean 仍然有差距。

## 11. 失败样本分析

为了不只靠主观听感，我新增了失败模式分析工具：

```text
scripts/analyze_failure_modes.py
```

它会分析：

- 语音保持程度。
- 静音段残留噪声。
- 低频风噪残留。
- 高频 hiss。
- SI-SDR regression。

wind eval 上的结果：

| failure mode | 数量 |
| --- | ---: |
| ok | 117 |
| low_freq_wind | 35 |
| regression | 7 |
| residual_noise | 1 |

关键指标：

| 指标 | 数值 |
| --- | ---: |
| mean speech preservation | -0.96 dB |
| mean quiet noise reduction | 15.73 dB |
| mean quiet low RMS | -40.45 dBFS |
| mean quiet high RMS | -71.64 dBFS |

这个分析说明：当前主要问题不是人声整体被压小，而是仍有一部分样本存在低频风噪/气流残留。

DEMAND eval 上：

| failure mode | 数量 |
| --- | ---: |
| ok | 117 |
| low_freq_wind | 26 |
| regression | 13 |
| residual_noise | 1 |
| speech_too_small | 3 |

这说明当前模型偏向风噪优化之后，对通用 DEMAND 噪声有一定退化。

## 12. 当前已完成的工程内容

目前已经完成：

- 数据准备和公开数据整理。
- 动态双麦混音训练集。
- Tiny band-mask TCN baseline。
- Tiny DeepFilter TCN。
- coherence/MWF 空间前端。
- MWF teacher 蒸馏训练。
- wind-noise fine-tune。
- 多种后处理和对比模型实验。
- 离线评估脚本。
- 实时增强脚本。
- 网页试听 demo。
- 失败样本分析脚本。
- 多份 README / report 文档。
- 早期 INT8 导出和 C reference 链路。
- GitHub 仓库同步。

## 13. 当前不足

当前项目还存在几个主要不足：

1. **和 clean 仍有听感差距**

   虽然当前版本有降噪效果，但低频风噪/呼呼声还没有完全消除。

2. **真实数据不足**

   当前训练仍然主要是公开 clean speech + noise 合成，和真实助听器佩戴位置、麦克风自噪声、风噪、摩擦噪声还有差距。

3. **风噪和通用噪声之间有取舍**

   当前 conservative teacher 版本提升了 wind eval，但 DEMAND eval 比普通 wind fine-tune 略低。

4. **最新 DeepFilter 主线还没完全走完端侧导出**

   早期 band-mask TCN 已经做了 INT8/C reference，但当前推荐的 Tiny DeepFilter teacher 版本还需要继续做 INT8 导出、fixed-scale 验证和 C reference 对齐。

5. **主观听感和 SI-SDR 不完全一致**

   后续需要更系统的听感评价和更贴近听感的指标。

## 14. 下一步计划

我建议下一步按以下优先级推进。

### 14.1 低频风噪定向优化

使用 failure analysis 中标记为 `low_freq_wind` 的样本作为固定失败集，专门优化低频风噪残留。

具体方向：

- 加低频 residual wind loss。
- 对静音/弱语音区低频残留加约束。
- 避免全频强压噪导致人声损伤。

### 14.2 语音保持约束

虽然当前人声平均只低约 0.96 dB，但后续继续压风噪时可能再次损伤语音，所以要加入 speech preservation 约束。

方向：

- speech-active RMS preservation。
- speech band loss。
- 对高 SNR 样本减少过处理。

### 14.3 真实双麦数据补充

如果要进一步接近真实助听器效果，需要补真实设备数据。

建议采集：

- 佩戴位置双麦语音。
- 风噪。
- 走动气流噪声。
- 摩擦噪声。
- 不同房间混响。

哪怕少量真实数据，也可以用于 fine-tune 和 failure set。

### 14.4 当前主线端侧部署闭环

把当前推荐 Tiny DeepFilter teacher 版本继续做工程化：

- INT8 导出。
- fixed-scale Python reference。
- C reference 对齐。
- benchmark 模型状态和每帧耗时。
- 后续替换 CMSIS-DSP / CMSIS-NN。

### 14.5 展示和评估完善

- 网页 demo 继续保持当前推荐版本在最前。
- 增加更多固定风噪样本。
- 汇报中同时展示 SI-SDR、失败模式分析和主观试听。
- 不只用单一指标判断模型好坏。

## 15. 汇报时可以重点强调的结论

可以用下面几句话总结当前项目进展：

```text
目前项目已经完成了从数据准备、模型训练、实时增强、网页试听到失败样本分析的闭环。

最初的 121K band-mask TCN 能实现基本降噪，但底噪和呼呼声明显。

经过空间特征、DeepFilter、coherence/MWF 前端、风噪数据和 MWF teacher 蒸馏后，当前主线模型为 137,984 参数的 TinyDeepFilterTCN。

当前 conservative MWF teacher 版本在 Zenodo wind eval 上达到 +4.80 dB SI-SDR improvement，比普通风噪微调版的 +4.35 dB 更好。

失败样本分析显示，现在主要瓶颈不是人声整体过小，而是部分样本的低频风噪残留。

下一步重点应该围绕低频风噪失败样本、真实双麦数据和当前 DeepFilter 主线的端侧部署闭环继续推进。
```

## 16. 汇报建议顺序

正式和老师讲时，建议按这个顺序：

1. 先说明项目目标：双麦端侧助听器/耳戴设备降噪，小模型实时。
2. 说明为什么不是普通单麦降噪：双麦有空间信息，但近距双麦更依赖 IPD/coherence。
3. 讲最初 baseline：121K band-mask TCN，完成了基本闭环但听感有问题。
4. 讲后续优化：DeepFilter、coherence/MWF 前端、风噪数据。
5. 重点讲当前版本：conservative MWF teacher distillation。
6. 解释 teacher 不是部署模型，只是训练阶段指导小模型。
7. 展示当前指标和网页 demo。
8. 展示失败样本分析：主要问题变成低频风噪残留。
9. 最后讲下一步：低频风噪定向优化、真实数据、端侧 INT8/C reference。

## 17. 可展示材料清单

汇报时可以打开：

```text
README.md
docs/current_progress_summary.md
docs/conservative_mwf_teacher_deepfilter.md
docs/teacher_meeting_report.md
runs/arctic_wind_teacher_deepfilter_mwf_conservative/analysis_wind_zenodo/failure_analysis.md
```

网页 demo：

```text
http://127.0.0.1:38180/runs/audio_demo/index.html
```

优先播放：

```text
teacher_deepfilter_conservative_on_wind_eval
```

对比播放：

```text
wind_finetune_on_wind_eval
teacher_deepfilter_aggressive_on_wind_eval
```
