# TinyHear 双麦端侧降噪版本演进报告

## 1. 报告目的

本报告整理 TinyHear Dual-Mic Denoising 项目从最初原型到后续多轮优化的完整演进过程，重点说明每一版为什么做、具体怎么实现、指标如何变化、主观听感暴露了什么问题，以及这些结果对后续方向的判断。

项目最初目标是：在助听器/耳戴设备这类端侧场景中，用双麦输入实现上行语音降噪，模型大小控制在 100-150 KB 左右，并尽量支持整型推理和实时运行。

后续实验的主线可以概括为：

```text
Tiny band-mask TCN
  -> 更真实公开数据集
  -> 空间特征和实时链路
  -> learned gate / bypass
  -> residual / DeepFilter
  -> coherence-weighted spatial frontend
  -> stable postfilter / dehiss
  -> RNNoise-style TinyGRU
  -> full-bin complex mask TCN
```

整体结论是：项目已经完成了数据、训练、量化、C reference、网页 demo 和多种模型原型闭环；但从主观听感看，当前公开合成双麦数据和小模型路线仍没有彻底解决“底噪大、呼呼声明显、人声不够自然”的问题。尤其是后处理和简单 mask 方向的收益已经接近上限。

## 2. 统一实验设置

### 2.1 音频与实时约束

基础 DSP 配置：

| 项目 | 设置 |
| --- | ---: |
| 采样率 | 16 kHz |
| STFT | 256 点 FFT |
| hop | 64 samples |
| 帧步进 | 4 ms |
| 输出 | 单通道增强语音 |
| 模型约束 | 100-150 KB 级别 |
| 因果性 | 模型不使用未来帧 |

早期 realtime 链路还验证了：

- Python streaming 模型和 offline 模型输出几乎一致。
- C reference 中 Q15 streaming 输出和 batch Q15 输出一致。
- PC reference 上完整 realtime DSP 约 `2.5 ms/hop`，低于 4 ms hop 预算。

### 2.2 数据集

项目先后使用过：

| 阶段 | clean speech | noise | 用途 |
| --- | --- | --- | --- |
| synthetic baseline | 合成语音 | 合成噪声 | 快速打通流程 |
| YESNO + DEMAND | OpenSLR YESNO | DEMAND 多通道噪声 | 公开真实语音/环境噪声 sanity check |
| ARCTIC + DEMAND | CMU ARCTIC | DEMAND 多通道噪声 | 当前主要训练和验证基线 |

当前主要评估集是固定生成的：

```text
data/arctic_demand_eval/val
```

包含 160 条 `mix_*.wav` 和对应 `clean_*.wav`。大多数后期版本都在这 160 条固定样本上对比，便于横向比较。

### 2.3 指标与听感

主要客观指标：

- `mean_noisy_si_sdr`
- `mean_enhanced_si_sdr`
- `mean_si_sdr_improvement`
- `mean_output_input_rms_ratio`

但是项目后期暴露出一个重要问题：**SI-SDR 不能完整代表主观听感**。有些版本 SI-SDR 高，但仍有明显“呼呼声/气流声”；有些版本 RMS 更接近输入，但人声听感并没有明显变好。因此后续报告中同时记录主观结论。

## 3. 版本总览

固定 160 条 ARCTIC + DEMAND 验证集上，部分代表版本指标如下：

| 版本 | 主要思路 | 参数量 | SI-SDR improvement | RMS ratio | 主观结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `arctic_demand` | 原始 Tiny band-mask TCN | 约 121K | +4.75 dB | - | 有基本降噪，但人声偏小、底噪明显 |
| `spatial_c116_loud` | 空间特征 + c116 + loudness/gate | - | +4.64 dB | 0.928 | 人声响度改善有限 |
| `c120_psm_gain_noise_loss` | PSM + noise loss | - | +5.16 dB | 0.994 | 指标略升，听感问题仍在 |
| `tiny_deepfilter_beamform` | DeepFilter + delay-sum | - | +9.39 dB | 0.994 | 指标明显提升 |
| `tiny_deepfilter_coherence_mwf` | coherence spatial frontend + DeepFilter | 137,984 | +9.46 dB | 0.992 | 指标最强之一，但仍有呼呼声 |
| `coherence_mwf_smooth` | 更平滑的 coherence frontend | - | +9.43 dB | 0.992 | 稳定性略好，呼呼声仍存在 |
| `stable_postfilter` | stationary noise floor postfilter | - | +9.24 dB | 0.992 | 更保守，但未根治气流感 |
| `deepfilter_dehiss` | 高频 residual dehiss | - | +9.38 dB | 0.914 | 高频压了一些，但听感仍不理想 |
| `deepfilter_dehiss_aggressive` | 更强 dehiss | - | +9.26 dB | 0.914 | 更安静但可能损伤自然度 |
| `tiny_gru_h136` | RNNoise-style GRU band gain | 142,561 | +4.77 dB | 0.994 | 指标和听感均不如 DeepFilter |
| `tiny_complex_mask` | full-bin complex mask TCN | 92,018 | +9.59 dB | 0.938 | 指标略高，但主观听感反馈不如预期 |

注意：`tiny_complex_mask` 的 SI-SDR 指标最高，但用户试听反馈“甚至不如之前几个版本，感觉跟没有降噪没有区别”。这进一步说明当前评价体系和真实主观目标之间存在偏差。

## 4. 初始 Tiny band-mask TCN

### 4.1 优化目标

最初目标是快速实现一个可训练、可导出、可实时运行的小模型。

基本路径：

```text
dual-mic waveform
  -> STFT
  -> 32-band 特征
  -> TinyCausalTCN
  -> 32-band mask
  -> mask 插值到 FFT bin
  -> 乘到参考麦克风频谱
  -> ISTFT
```

### 4.2 实现方式

核心模型：`TinyCausalTCN`

文件：

```text
src/ha_denoise/model.py
scripts/train.py
scripts/evaluate.py
scripts/enhance_realtime.py
```

模型结构：

```text
1x1 Conv + ReLU
8 个 causal depthwise block
1x1 mask head
hard-sigmoid 输出 band mask
```

早期特征为 96 维：

```text
mic0 log band power: 32
mic1 log band power: 32
ILD / log power ratio: 32
```

### 4.3 结果

在 ARCTIC + DEMAND 固定验证集上：

```text
mean_noisy_si_sdr:      4.33 dB
mean_enhanced_si_sdr:   9.08 dB
improvement:            4.75 dB
```

### 4.4 问题

这个版本具备基本降噪能力，但主观听感主要问题是：

- 底噪仍明显。
- 人声会变小。
- 干净样本上可能过处理。
- 只预测频带幅度 mask，无法真正处理 noisy phase 和细粒度残留噪声。

## 5. 数据升级：YESNO/ARCTIC + DEMAND

### 5.1 优化目标

合成数据过于简单，不能代表真实环境。因此引入公开语音和多通道环境噪声：

- YESNO + DEMAND：用于小规模 sanity check。
- CMU ARCTIC + DEMAND：作为后续主基线。

### 5.2 实现方式

相关脚本：

```text
scripts/prepare_demand_noise.py
scripts/prepare_wav_dataset.py
scripts/materialize_mixes.py
```

DEMAND 多通道噪声被切成 stereo noise chunk；训练时通过 `WavPairDataset(on_the_fly=True)` 动态混合 clean speech 和 noise。

双麦合成逻辑位于：

```text
src/ha_denoise/dataset.py
```

核心包括：

- clean 归一到固定 RMS。
- noise 按随机 SNR 混入。
- noise 使用随机入射角产生双麦时间差。
- 可选 RIR 卷积。
- 输出 `mix[2, samples]` 和 `clean_pair[2, samples]`。

### 5.3 结果与判断

YESNO + DEMAND 固定验证集：

```text
SI-SDR improvement: +6.02 dB
```

ARCTIC + DEMAND 固定验证集：

```text
SI-SDR improvement: +4.75 dB
```

ARCTIC 指标更低，但任务更真实，因为语音内容和噪声环境更丰富。后续所有核心优化都以 ARCTIC + DEMAND 作为主基线。

## 6. 空间特征：IPD 与 coherence

### 6.1 优化动机

双麦距离很近时，ILD 几乎没有明显差别，单纯使用左右能量差不够。用户明确提出：近距双麦下 IPD 应该更有效。

### 6.2 实现方式

在 `FeatureConfig(spatial_features=True)` 时，输入特征从 96 维扩展到 192 维：

```text
mic0 log power: 32
mic1 log power: 32
ILD: 32
IPD cos: 32
IPD sin: 32
coherence: 32
```

实现位置：

```text
src/ha_denoise/features.py
```

训练入口支持：

```bash
--spatial-features
```

### 6.3 效果

空间特征使模型可以利用双麦相位差和相干性，但 band-mask 主体结构仍然限制了能力。后续 `c116/c120`、PSM、noise loss 等版本都基于这一方向继续尝试。

代表结果：

```text
c120_psm_gain_noise_loss: +5.16 dB
c120_psm_gain_sisdr:     +5.10 dB
```

这些比最早 ARCTIC baseline 略高，但仍远低于后续 DeepFilter。

## 7. 实时链路、INT8 与 C reference

### 7.1 优化目标

项目不是只做离线 Python 模型，而是要证明它有端侧部署路径。因此补齐：

- Python streaming inference。
- INT8 权重导出。
- activation scale calibration。
- C Q15/int8 reference。
- C realtime DSP reference。
- PC benchmark。

### 7.2 实现方式

相关文件：

```text
scripts/export_int8.py
scripts/calibrate_int8.py
scripts/verify_int8_reference.py
scripts/dump_c_reference_assets.py
c_reference/
```

C reference 包括：

```text
tiny_tcn_forward
tiny_tcn_forward_q15
tiny_tcn_process_frame_q15
tiny_realtime_process_hop
fft_backend.h/.c
```

### 7.3 结果

早期 reference 验证：

```text
stream_mismatches: 0
model_ms_per_frame: about 0.30 ms
dsp_ms_per_hop: about 2.5 ms
realtime_budget: 4.0 ms/hop
```

这说明工程链路是完整的，但该阶段主要解决“能部署、能实时”，没有解决主观呼呼声问题。

## 8. high-SNR bypass 与 learned gate

### 8.1 优化动机

用户试听发现部分样本已有一定 SNR，模型仍然处理，导致：

- 人声变小。
- 干净片段被破坏。
- 噪声和人声响度比例不自然。

### 8.2 实现方式

先实现 heuristic bypass：

```bash
--high-snr-bypass
--bypass-threshold
--bypass-width
```

后续实现 learned gate：

```text
scripts/train_gate.py
scripts/evaluate_learned_gate.py
scripts/export_gate.py
```

gate 思路是学习一个输入相关的混合权重，在模型输出和原始/beamformed 输入之间做动态 blend，降低过处理风险。

### 8.3 结果与问题

gate/bypass 能改善“人声太小”的一部分问题，但没有真正解决底噪和呼呼声。原因是它主要控制处理强度，而不是提升噪声建模能力。

典型版本：

```text
arctic_demand_spatial_c116_loud: +4.64 dB, RMS ratio 0.928
```

听感上人声响度略可控，但噪声残留仍明显。

## 9. residual noise loss / PSM / 训练目标改进

### 9.1 优化动机

单纯 mask MSE 不等价于听感好。于是加入：

- phase-sensitive mask target。
- band magnitude loss。
- SI-SDR loss。
- residual noise loss。
- silence floor loss。

### 9.2 实现方式

主要在：

```text
scripts/train.py
scripts/train_deepfilter.py
src/ha_denoise/features.py
```

PSM target 的核心思想：

```text
mask = Re(clean * conj(noisy)) / |noisy|^2
```

residual noise loss 的思路是：在 clean 能量低的频点或帧，对 enhanced 残留谱能量施加额外惩罚。

### 9.3 结果

PSM/noise-loss 分支在 band-mask TCN 上有小幅提升：

```text
c120_psm_gain:            +5.14 dB
c120_psm_gain_sisdr:      +5.10 dB
c120_psm_gain_noise_loss: +5.16 dB
```

但主观问题没有根本变化。判断是：band-mask 输出本身过粗，训练目标改进无法突破结构上限。

## 10. DeepFilter 分支

### 10.1 优化动机

band mask 只能按频带缩放，无法对复数谱做更细粒度修正。于是引入轻量 DeepFilter：在 ERB band gain 之外，对低频若干 FFT bins 加一个小阶数复数滤波残差。

### 10.2 实现方式

模型：

```text
TinyDeepFilterTCN
```

位置：

```text
src/ha_denoise/model.py
src/ha_denoise/features.py
scripts/train_deepfilter.py
scripts/evaluate_deepfilter.py
scripts/enhance_deepfilter.py
```

输出包括两部分：

```text
band_gain: [T, bands]
df_coef:   [T, df_bins, df_order, 2]
```

重构逻辑：

```text
enhanced = noisy_spec * band_gain
low_bins += sum_k complex_coef[k] * noisy_spec[t-k]
```

这相当于在低频区域引入短时复数滤波能力，同时保留小模型规模。

### 10.3 结果

DeepFilter 是整个项目中第一次显著提升 SI-SDR 的版本：

| 版本 | SI-SDR improvement | RMS ratio |
| --- | ---: | ---: |
| `tiny_deepfilter_beamform` | +9.39 dB | 0.994 |
| `tiny_deepfilter_coherence_mwf` | +9.46 dB | 0.992 |
| `coherence_mwf_smooth` | +9.43 dB | 0.992 |

### 10.4 主观问题

虽然指标明显提升，但用户反馈仍然存在：

- 底噪偏大。
- 类似气流的“呼呼声”。
- 和 clean 相比差距明显。

这说明 DeepFilter 的低频复数滤波和 band gain 仍不能充分处理全频段残留噪声，尤其是高频宽带残留。

## 11. coherence-weighted spatial frontend

### 11.1 优化动机

delay-and-sum 只做时延对齐和平均，不能利用双麦间短时相干性。对于非目标方向或扩散噪声，双麦相干性通常低于目标语音。因此加入 coherence-weighted spatial frontend。

### 11.2 实现方式

实现位置：

```text
src/ha_denoise/spatial.py
```

核心步骤：

```text
1. estimate_relative_delay
2. delay align mic1
3. delay-and-sum
4. STFT(mic0), STFT(aligned mic1)
5. smooth cross coherence
6. gain = floor + (1-floor) * coherence^gamma
7. apply gain to summed spectrum
```

支持模式：

```text
delay_sum
coherence_mwf
coherence_mwf_smooth
```

### 11.3 结果

coherence frontend 和 DeepFilter 组合后，客观指标达到较高水平：

```text
tiny_deepfilter_coherence_mwf: +9.46 dB
coherence_mwf_smooth:         +9.43 dB
```

### 11.4 问题

coherence frontend 可以降低部分空间噪声，但它仍是规则前端，不是完整的 MVDR/BF，也不会生成干净语音相位。对于“呼呼声”这种宽带残留或调制噪声，它只能缓解，不能根治。

## 12. stable postfilter 和 dehiss

### 12.1 优化动机

用户多次反馈“底噪还是大”“有气流呼呼声”。在不重新训练模型的前提下，尝试后处理：

- stationary noise floor filter。
- high-frequency residual dehiss。
- aggressive dehiss。

### 12.2 实现方式

实现位置：

```text
src/ha_denoise/features.py
```

主要函数：

```text
stationary_noise_floor_filter
residual_dehiss_filter
```

`stationary_noise_floor_filter`：

- 从整段音频的低分位数谱能量估计稳定噪声底。
- 计算 Wiener-like gain。
- 用模型 band mask 估计 speech presence，保护语音区域。
- 对 gain 做时间平滑，避免 pumping。

`residual_dehiss_filter`：

- 主要压制 2.6 kHz 以上高频残留。
- 根据高频权重降低 floor。
- 用 speech presence 保护疑似语音区域。
- 提供 aggressive 参数组，更强压制高频残留。

### 12.3 结果

| 版本 | SI-SDR improvement | RMS ratio |
| --- | ---: | ---: |
| `stable_postfilter` | +9.24 dB | 0.992 |
| `deepfilter_dehiss` | +9.38 dB | 0.914 |
| `deepfilter_dehiss_aggressive` | +9.26 dB | 0.914 |

### 12.4 结论

后处理能改变噪声响度和频谱倾向，但没有解决根因。aggressive 版本更安静，但会牺牲自然度；保守版本自然度更好，但呼呼声仍在。后处理路线已经接近上限。

## 13. RNNoise-style TinyGRU

### 13.1 优化动机

RNNoise 这类模型使用 GRU 做时序建模，理论上可以降低 frame-to-frame 抖动和音乐噪声。项目尝试一个 150K 内的 GRU band-gain 模型，验证时序连续性是否能改善呼呼声。

### 13.2 实现方式

模型：

```text
TinyGRUDenoiser
```

文件：

```text
src/ha_denoise/model.py
scripts/train_tiny_gru.py
scripts/evaluate_tiny_gru.py
```

结构：

```text
Linear feature projection
GRU(hidden=136)
band gain head
VAD head
```

参数量：

```text
hidden=96:  77,601
hidden=136: 142,561
```

最终保留 h136 版本。

### 13.3 结果

```text
tiny_gru_h136:
items: 160
SI-SDR improvement: +4.77 dB
RMS ratio: 0.994
```

### 13.4 结论

TinyGRU 没有解决问题，原因是它仍然主要是 band gain + noisy/beamformed phase 路线。GRU 能平滑时间变化，但不能补足频谱和相位建模能力。

## 14. oracle 诊断

### 14.1 为什么做 oracle

多轮后处理和模型改动后，主观呼呼声仍然存在。因此需要判断问题来自：

```text
模型没学好
```

还是来自：

```text
当前 mask/noisy-phase 路线本身上限不足
```

### 14.2 诊断结果

使用 oracle band mask + noisy phase：

```text
mean_si_sdr_improvement: +5.64 dB
```

这个结果非常关键：即使知道 clean 幅度目标，只要仍然使用 noisy phase 和粗 band mask，上限也不高。

### 14.3 结论

这解释了为什么 TinyGRU 和早期 band-mask 版本很难解决呼呼声。呼呼声不只是参数问题，而是增强形式的问题：

- band mask 太粗。
- noisy phase 保留了很多噪声结构。
- 高频残留无法靠简单 gain 完全消除。

## 15. full-bin complex mask TCN

### 15.1 优化动机

为了突破 band-mask 上限，新增 full-bin complex mask 模型。它不再只预测 32-band gain，而是预测每个 FFT bin 的复数 mask：

```text
real_mask[f, t]
imag_mask[f, t]
```

### 15.2 实现方式

模型：

```text
TinyComplexMaskTCN
```

文件：

```text
src/ha_denoise/model.py
src/ha_denoise/features.py
scripts/train_complex_mask.py
scripts/evaluate_complex_mask.py
```

输出：

```text
[T, 129, 2]
```

重构：

```text
enhanced_spec = noisy_or_beamformed_spec * complex_mask
```

训练目标包括：

```text
complex mask MSE
waveform L1 loss
complex STFT loss
log magnitude loss
SI-SDR loss
high-band residual loss
```

模型初始化时将 real mask bias 设为接近 1，imag mask 设为 0，使初始输出接近直接通过 beamformed 输入，避免一开始输出静音。

参数量：

```text
92,018
```

训练记录：

```text
best epoch: 5
val_loss: 14.5719
```

### 15.3 客观结果

固定 160 条验证集：

```text
mean_noisy_si_sdr:      4.33 dB
mean_enhanced_si_sdr:   13.93 dB
improvement:            +9.59 dB
RMS ratio:              0.938
```

这是当前客观 SI-SDR 最高的版本之一。

### 15.4 主观结果

用户试听反馈：

```text
tiny_complex_mask 甚至不如之前几个版本，感觉跟没有降噪没有区别。
```

这说明当前 complex mask 虽然方向更合理，但第一版训练还没有把主观听感做出来。可能原因：

- 训练时间短，只训到 epoch 5 附近，模型仍欠训练。
- 92K 参数可能偏小，频点级输出学习难度高。
- 当前 loss 仍偏向 SI-SDR/STFT，而不是专门针对噪声残留主观感知。
- ARCTIC + DEMAND 合成方式和真实听感需求仍有 mismatch。
- 网页 5 条样例可能不能代表指标均值。

因此这一版不能作为最终效果版本，只能说明项目已经切换到更正确的建模方向。

## 16. 网页 demo 与听感评估

### 16.1 优化动机

最初每次都要下载 wav 自己听，效率很低。因此实现网页 demo。

### 16.2 实现方式

脚本：

```text
scripts/build_audio_demo.py
```

输出：

```text
runs/audio_demo/index.html
```

网页展示多组版本，每组包含：

```text
Noisy
Offline
Realtime
Clean
```

并显示每条样例的 SI-SDR improvement。

### 16.3 当前推荐比较顺序

当前网页中保留多个历史版本，便于 A/B：

```text
tiny_complex_mask
deepfilter_dehiss_aggressive
deepfilter_dehiss
tiny_gru_h136
stable_postfilter
coherence_mwf_smooth
tiny_deepfilter_coherence_mwf
...
```

实际听感比较时，不应只看第一组，而应重点比较：

```text
tiny_deepfilter_coherence_mwf
coherence_mwf_smooth
stable_postfilter
deepfilter_dehiss
tiny_complex_mask
```

## 17. 为什么多版仍没有彻底解决呼呼声

综合所有实验，目前判断如下。

### 17.1 后处理不是根因解

stable postfilter 和 dehiss 能压一部分噪声，但会在以下两者之间摇摆：

```text
更安静，但人声变闷/不自然
更自然，但呼呼声仍在
```

因此不能靠继续调参数根治。

### 17.2 band mask 路线有理论上限

oracle band mask + noisy phase 只有 `+5.64 dB`，说明只做 band gain 很难获得 clean-like 输出。

### 17.3 SI-SDR 与听感不一致

DeepFilter 和 complex mask 都能获得约 `+9 dB` 的 SI-SDR improvement，但用户仍觉得底噪和呼呼声明显。这说明需要额外指标：

- 高频 residual noise ratio。
- non-speech frame attenuation。
- speech-band RMS preservation。
- PESQ/STOI/DNSMOS 等感知指标。
- 主观 ABX/MOS 测试。

### 17.4 数据 mismatch 是核心风险

当前数据是公开 clean speech + DEMAND 噪声合成。它不等价于真实助听器/耳戴设备双麦录音。真实场景中会有：

- 设备壳体遮挡。
- 近场说话人。
- 佩戴位置变化。
- 麦克风频响不一致。
- 机械摩擦/风噪。
- 房间混响和头部散射。

这些都可能导致模型在合成验证集上指标高，但实际听感不好。

## 18. 当前代码与产物

关键代码：

| 文件 | 作用 |
| --- | --- |
| `src/ha_denoise/model.py` | TinyCausalTCN、TinyDeepFilterTCN、TinyGRUDenoiser、TinyComplexMaskTCN |
| `src/ha_denoise/features.py` | STFT、特征、mask target、DeepFilter、complex mask、postfilter |
| `src/ha_denoise/spatial.py` | delay-and-sum、coherence spatial frontend |
| `src/ha_denoise/dataset.py` | clean/noise 动态混音和双麦合成 |
| `scripts/train.py` | band-mask TCN 训练 |
| `scripts/train_deepfilter.py` | DeepFilter 训练 |
| `scripts/train_tiny_gru.py` | RNNoise-style GRU 训练 |
| `scripts/train_complex_mask.py` | complex mask TCN 训练 |
| `scripts/evaluate_*.py` | 各版本评估 |
| `scripts/build_audio_demo.py` | 网页 demo 生成 |
| `c_reference/` | C reference 和 benchmark |

关键产物：

| 目录 | 内容 |
| --- | --- |
| `runs/arctic_demand` | 原始 ARCTIC + DEMAND baseline |
| `runs/arctic_demand_spatial_c116_loud` | loud/gate 相关版本 |
| `runs/arctic_demand_tiny_deepfilter_coherence_mwf` | DeepFilter 主力版本 |
| `runs/arctic_demand_tiny_deepfilter_stable_postfilter` | stable postfilter demo |
| `runs/arctic_demand_tiny_gru_h136` | RNNoise-style TinyGRU |
| `runs/arctic_demand_tiny_deepfilter_dehiss*` | dehiss 后处理版本 |
| `runs/arctic_demand_tiny_complex_mask` | full-bin complex mask 版本 |
| `runs/audio_demo/index.html` | 网页试听 demo |

## 19. 阶段性结论

项目已经完成了一个比较完整的端侧双麦 AI 降噪研究原型：

1. 数据准备、训练、验证、听感 demo 已打通。
2. 端侧约束下的 Tiny TCN、DeepFilter、GRU、complex mask 都已实现。
3. C reference、INT8/Q15、realtime DSP 链路已经建立。
4. 多轮版本清楚暴露出：客观 SI-SDR 提升不等于主观听感提升。

从当前结果看，最稳的客观版本仍是 DeepFilter/coherence 系列；`tiny_complex_mask` 是更合理的研究方向，但第一版主观表现不稳定，不应作为最终效果结论。

## 20. 后续建议

如果后续继续优化，建议不要再做简单后处理调参，而是按下面顺序推进：

1. 建立更可靠的主观/感知评估集：固定 20-50 条样例，记录每版 ABX 结果。
2. 引入真实或更接近真实的耳戴双麦数据，哪怕只有少量，用于验证和微调。
3. 对 DeepFilter/coherence 主线做 teacher-student 蒸馏，而不是从头小模型硬学。
4. complex mask 方向需要更长训练、更大 teacher、更强感知 loss，再压缩到 100-150K。
5. 增加 DNSMOS/PESQ/STOI 和高频 residual 指标，避免只被 SI-SDR 误导。
6. 如果继续限制 150K，优先保留 DeepFilter 主线；如果可以放宽模型大小，再考虑更强的 causal Conv-TasNet/TF-GridNet-lite/DeepFilterNet-lite teacher。

本阶段最重要的经验是：项目工程闭环已经比较完整，但“呼呼声”属于主观噪声质量问题，不能只靠 SI-SDR 和后处理解决。下一阶段应转向真实数据、感知指标和 teacher-student 训练。
