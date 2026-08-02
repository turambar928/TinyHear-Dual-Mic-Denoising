# Conservative MWF Teacher DeepFilter

本文档说明当前风噪优化版本的完整实现方式。这个版本不是直接部署 oracle MWF，而是把 oracle local MWF 当作训练阶段的 teacher，用较小权重蒸馏到 137,984 参数的 Tiny DeepFilter TCN 中。

## 目标

项目目标是做一个适合助听器/耳戴设备上行链路的双麦端侧降噪模型：

- 双麦输入，利用近距离麦克风之间的相位差、相干性和空间信息。
- 模型大小控制在 100-150K 参数级别。
- 推理链路适合 16 kHz、4 ms hop 的实时处理。
- 后续可导出 INT8 权重，端侧用 int8 卷积、int32 累加实现。
- 当前重点解决听感里的底噪、风噪和类似气流的“呼呼声”。

## 为什么要加 MWF Teacher

之前尝试过多条路线：

- 纯频带 mask：能降噪，但容易留下背景呼呼声。
- postfilter/dehiss：能压一点高频残留，但根因没有解决。
- TinyGRU/RNNoise-style：模型轻，但在当前双麦风噪任务上能力不足。
- complex-mask/student：没有稳定改善听感。
- wind-noise fine-tune：在真实风噪数据上有提升，但离 clean 仍有明显差距。

后续用 oracle local MWF 做上限验证时，发现强双麦前端在风噪 eval 上明显更强：

```text
Oracle local MWF on Zenodo wind eval:
Noisy SI-SDR:     5.39 dB
Enhanced SI-SDR: 14.21 dB
Improvement:     +8.82 dB
```

这说明方向不是继续堆后处理，而是让小模型学习更接近 MWF 的滤波行为。但 oracle MWF 依赖 clean/noise 信息，不能直接部署，所以只在训练阶段作为 teacher。

## 数据

当前训练数据由三部分组成：

- clean speech：CMU ARCTIC。
- 普通环境噪声：DEMAND 多通道环境噪声。
- 风噪：Zenodo Wind Noise Dataset。

本地整理后的训练目录：

```text
data/arctic_wind_demand_flat/
  train/
    clean/
    noise/
  val/
    clean/
    noise/
```

固定评估集：

```text
data/arctic_wind_eval/      # Zenodo wind-noise eval
data/arctic_demand_eval/    # original DEMAND eval
```

原始下载数据和整理后的大数据目录被 `.gitignore` 忽略，不提交到 GitHub；只提交 checkpoint、metrics 和少量网页试听样本。

## 模型结构

当前推荐模型：

```text
runs/arctic_wind_teacher_deepfilter_mwf_conservative/best.pt
```

结构参数：

```text
model_type: tiny_deepfilter_tcn
parameters: 137,984
sample_rate: 16 kHz
n_fft: 256
hop_length: 64
hop_duration: 4 ms
channels: 96
blocks: 8
kernel_size: 5
df_bins: 64
df_order: 3
spatial_frontend: coherence_mwf
```

模型输入不是单麦幅度谱，而是双麦空间特征加上前端参考通道特征。模型输出两类参数：

- `gain`：频带增益，用于整体谱幅度抑制。
- `coef`：低频 deep-filter 复数系数，用于多帧复数域滤波，减少只做幅度 mask 带来的相位/音乐噪声问题。

## 推理流程

推理时没有 clean，也没有 oracle teacher。实际增强流程如下：

1. 读取双通道 mixture wav。
2. 对双麦做 STFT。
3. 计算双麦空间特征，包括相位差、相干性等。
4. 使用 `coherence_mwf` 空间前端生成参考通道。
5. Tiny DeepFilter TCN 逐帧预测 `gain` 和 `coef`。
6. 对参考通道做 band gain + deep filtering。
7. iSTFT / overlap-add 还原增强语音。
8. 可选 dehiss postfilter 用于网页试听评估。

网页 demo 中的 `Realtime` 表示按实时链路生成的增强结果，`Clean` 是目标干净语音，`Noisy` 是输入混合语音。

## Teacher 训练流程

训练阶段额外打开：

```bash
--teacher-mwf
```

此时 dataloader 会返回：

- `mix_refs`：模型实际参考通道。
- `clean_refs`：clean mic0 目标。
- `mix_pairs`：双麦 mixture。
- `clean_pairs`：双麦 clean。

训练脚本会为每个 batch 在线计算 oracle local MWF teacher：

1. 对 `mix_pairs` 和 `clean_pairs` 做双麦 STFT。
2. 用局部时间窗口估计二阶统计量。
3. 计算双通道局部 MWF 滤波器。
4. 用该滤波器处理 mixture，得到 teacher waveform。
5. 将 teacher waveform 与 clean mic0 混合，形成训练目标。

当前保守版本使用：

```text
oracle_window: 9
diagonal_loading: 0.01
teacher_blend: 0.25
teacher_mask_weight: 0.35
```

目标波形为：

```text
target = 0.25 * oracle_mwf_teacher + 0.75 * clean_mic0
```

这样做的原因是：teacher 太强会把小模型拉向过度滤波，导致人声变小、通用噪声场景退化。我们也训练过一个激进版本：

```text
teacher_blend: 0.90
teacher_mask_weight: 0.75
```

结果风噪 eval 只有 `+3.74 dB`，低于普通 wind fine-tune 的 `+4.35 dB`，说明小模型不能硬学 oracle MWF。

## Loss 设计

训练 loss 由多项组成：

- `masked_mse`：预测 band gain 与目标 mask 的误差。
- `waveform_l1_loss`：增强波形与目标波形的 L1 误差。
- `stft_logmag_loss`：log magnitude 谱误差，约束听感相关的谱幅度。
- `si_sdr_loss`：提升整体语音重建质量。
- `coef_energy_loss`：限制 deep-filter 复数系数过大，减少不稳定伪影。
- `residual_noise_loss`：对 clean 中能量较低的频点施加残留噪声惩罚。
- `silence_floor_loss`：压低静音和弱语音区域的输出底噪。

保守 teacher 训练命令中的关键权重：

```text
waveform_loss_weight: 0.70
stft_mag_loss_weight: 0.22
si_sdr_loss_weight: 0.04
coef_reg_weight: 0.002
residual_noise_loss_weight: 0.34
silence_floor_weight: 0.28
```

## 训练命令

```bash
PYTHONPATH=src python3 scripts/train_deepfilter.py \
  --data data/arctic_wind_demand_flat \
  --out runs/arctic_wind_teacher_deepfilter_mwf_conservative \
  --resume runs/arctic_wind_demand_tiny_deepfilter_coherence_mwf/best.pt \
  --reset-best-on-resume \
  --epochs 4 \
  --batch-size 8 \
  --seconds 1.0 \
  --on-the-fly \
  --virtual-multiplier 2 \
  --snr-min-db -12 \
  --snr-max-db 8 \
  --noise-mix-prob 0.65 \
  --mic-distance-min-m 0.015 \
  --mic-distance-max-m 0.024 \
  --self-noise-prob 0.45 \
  --self-noise-db -42 \
  --wind-noise-prob 0.10 \
  --wind-noise-db -32 \
  --spatial-frontend coherence_mwf \
  --teacher-mwf \
  --teacher-mask-weight 0.35 \
  --teacher-blend 0.25 \
  --oracle-window 9 \
  --diagonal-loading 0.01 \
  --channels 96 \
  --blocks 8 \
  --kernel-size 5 \
  --df-bins 64 \
  --df-order 3 \
  --waveform-loss-weight 0.70 \
  --stft-mag-loss-weight 0.22 \
  --si-sdr-loss-weight 0.04 \
  --coef-reg-weight 0.002 \
  --residual-noise-loss-weight 0.34 \
  --residual-noise-threshold 0.075 \
  --silence-floor-weight 0.28 \
  --silence-threshold 0.04 \
  --min-gain 0.005 \
  --lr 1e-4 \
  --device cuda
```

训练结果：

```text
epoch=1 train_loss=0.527185 val_loss=0.507915
epoch=2 train_loss=0.515986 val_loss=0.498970
epoch=3 train_loss=0.515398 val_loss=0.513453
epoch=4 train_loss=0.506777 val_loss=0.496663
```

最终 `best.pt` 来自第 4 个 epoch。

## 评估结果

评估命令使用同一套 dehiss postfilter 参数，便于和上一版风噪模型横向比较。

```bash
PYTHONPATH=src python3 scripts/evaluate_deepfilter.py \
  --checkpoint runs/arctic_wind_teacher_deepfilter_mwf_conservative/best.pt \
  --data data/arctic_wind_eval \
  --split val \
  --max-items 160 \
  --save-audio runs/arctic_wind_teacher_deepfilter_mwf_conservative/eval_wind_zenodo \
  --save-listening runs/arctic_wind_teacher_deepfilter_mwf_conservative/listening_eval_wind_zenodo \
  --listening-samples 20 \
  --device cuda \
  --dehiss-postfilter \
  --dehiss-strength 1.05 \
  --dehiss-low-floor 0.82 \
  --dehiss-high-floor 0.42
```

结果：

| Eval set | Noisy SI-SDR | Enhanced SI-SDR | Improvement |
| --- | ---: | ---: | ---: |
| Zenodo wind eval | 5.39 dB | 10.19 dB | +4.80 dB |
| Original DEMAND eval | 4.33 dB | 10.96 dB | +6.63 dB |

对比上一版：

| Model | Wind eval improvement | DEMAND eval improvement |
| --- | ---: | ---: |
| wind fine-tune | +4.35 dB | +7.06 dB |
| aggressive MWF teacher | +3.74 dB | +5.21 dB |
| conservative MWF teacher | +4.80 dB | +6.63 dB |

结论：保守 teacher 版本对风噪更有效，但通用 DEMAND 噪声略有下降。它适合作为当前风噪/呼呼声问题的主线版本。

## 网页试听

启动服务：

```bash
python3 -m http.server 38180 --bind 0.0.0.0
```

打开：

```text
http://127.0.0.1:38180/runs/audio_demo/index.html
```

优先听：

```text
teacher_deepfilter_conservative_on_wind_eval
```

对比听：

```text
wind_finetune_on_wind_eval
teacher_deepfilter_aggressive_on_wind_eval
teacher_deepfilter_conservative_on_demand_eval
```

## 当前不足

这一版已经比普通 wind fine-tune 好一些，但和 clean 仍有差距，主要问题是：

- 风噪仍会以低频/宽带呼呼声形式残留。
- 强压噪会让人声响度和自然度下降。
- 公开数据集合成的双麦风噪与真实佩戴式设备仍有差距。
- SI-SDR 提升不完全等价于听感改善。

## 后续方向

更靠谱的下一步不是继续调 postfilter，而是：

- 补真实设备双麦风噪数据，哪怕少量也有价值。
- 做 teacher-student 的分阶段训练：先 clean target，再低权重 MWF teacher，再用听感样本微调。
- 增加更贴近风噪的评价指标，例如低频 residual energy、speech RMS preservation、segmental SNR。
- 尝试更明确的 speech-preservation loss，避免人声被压小。
- 做 INT8 导出和 C reference 对齐，把当前 DeepFilter 主线真正走到端侧部署。
