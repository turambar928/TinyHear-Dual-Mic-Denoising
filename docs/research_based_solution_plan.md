# 基于调研的 TinyHear 后续解决方案

## 1. 当前问题判断

当前项目已经尝试过 band mask、空间特征、DeepFilter、coherence 前端、postfilter、dehiss、TinyGRU 和 full-bin complex mask。客观指标最高版本能到 `+9 dB` 以上 SI-SDR improvement，但用户试听仍然反馈：

- 底噪大。
- 有类似气流的呼呼声。
- 人声不够自然。
- 某些高分版本主观上不如旧版本。

因此后续不应继续做简单后处理调参。更合理的方向是按照文献和实验结论，升级为：

```text
强双麦前端
  + 复数域/多帧滤波
  + 更贴近听感的 loss
  + 更接近真实设备的数据
```

## 2. 路线一：先把双麦前端做强

### 目标

当前 delay-and-sum 和 coherence-weighted frontend 只是轻量规则前端。下一步应该评估更接近 MWF/MVDR 的双麦前端上限。

### 已实现诊断

新增脚本：

```text
scripts/evaluate_oracle_mwf.py
```

它用已知 clean 估计局部双通道 Wiener 滤波器，属于 oracle/teacher 诊断，不是可部署算法。

用途：

```bash
PYTHONPATH=src python3 scripts/evaluate_oracle_mwf.py \
  --data data/arctic_demand_eval \
  --split val \
  --max-items 160 \
  --save-audio runs/oracle_mwf_arctic_demand/eval_audio \
  --save-listening runs/oracle_mwf_arctic_demand/listening_eval \
  --window 9
```

判断标准：

- 如果 oracle MWF 听起来明显接近 clean，说明强前端/teacher-student 路线值得继续。
- 如果 oracle MWF 仍然呼呼声大，说明问题更多来自数据合成、目标定义或评价方式。

当前固定 160 条 ARCTIC + DEMAND 验证集结果：

```text
mean_noisy_si_sdr:      4.33 dB
mean_enhanced_si_sdr:   20.39 dB
improvement:            +16.06 dB
```

这个结果明显高于当前所有小模型版本，说明“强双麦前端 + 多通道 Wiener/MWF teacher”是有效方向。下一步不应继续调 dehiss/postfilter，而应训练小模型去估计接近 oracle MWF 的 speech/noise mask 或滤波器。

### 后续可部署版本

oracle MWF 不能直接部署。实际部署可训练一个小模型估计：

```text
speech presence mask
noise mask
speech covariance / noise covariance
MWF/MVDR gain
```

然后在端侧执行轻量 2x2 复数矩阵滤波。

## 3. 路线二：不要只做幅度 mask，要做复数域

### 目标

呼呼声很可能来自 noisy phase 和高频残留。只做 magnitude/band mask 上限不足。

### 已尝试版本

`tiny_complex_mask` 已经实现 full-bin complex mask：

```text
real_mask[f, t]
imag_mask[f, t]
```

但第一版主观听感不好，说明“复数域方向正确”不等于“当前训练配置足够”。

### 下一步改法

不建议直接把 `tiny_complex_mask` 当最终版。更合理的是：

1. 先用 oracle MWF 或更大 teacher 生成 teacher enhanced。
2. 小模型学习 teacher 的 complex spectrum，而不是直接拟合 clean。
3. 输出形式可保留 complex mask，但 loss 主要对 teacher/clean 的谱和波形。

这样可以降低小模型直接学习干净复数谱的难度。

## 4. 路线三：多帧/复数滤波比单帧 mask 更适合 hearing aids

### 目标

hearing-aid 相关方法更常用多帧滤波，例如 MF-MVDR / MF-WF。这类方法不是只输出当前帧 gain，而是利用前后多帧构造滤波器。

### 可落地形式

端侧预算内可先做简化版：

```text
模型输出：
  speech mask
  noise mask
  或少量 filter coefficients

DSP 执行：
  causal covariance smoothing
  2x2 matrix inverse
  complex MWF filtering
```

这样模型不需要输出 129 个频点的完整复数 mask，也可能比单纯 band gain 更稳定。

## 5. 路线四：loss 不能只看 MSE 或 SI-SDR

### 当前问题

项目已经证明：

```text
SI-SDR 高 != 听感好
```

因此后续训练必须加入更贴近目标问题的指标。

### 推荐 loss 组合

```text
L = waveform SI-SDR
  + multi-resolution STFT loss
  + log-magnitude loss
  + high-frequency residual loss
  + speech-band preservation loss
  + noise-only frame attenuation loss
```

### 新增评估指标建议

后续网页和 metrics 应增加：

- `high_band_residual_ratio`: 2.5-8 kHz 残留/clean 比例。
- `speech_band_rms_ratio`: 300 Hz-3 kHz 人声能量保持。
- `quiet_frame_attenuation`: clean 低能帧上的输出抑制。
- `output_input_rms_ratio`: 已有，但要和 speech-band 分开看。

## 6. 路线五：数据比模型更重要

### 当前数据限制

当前数据是：

```text
CMU ARCTIC clean speech + DEMAND multichannel noise
```

这适合原型，但不等于真实助听器双麦数据。

真实呼呼声可能来自：

- 麦克风自噪声。
- 风噪。
- 佩戴摩擦。
- 头部遮挡。
- 麦克风频响不匹配。
- 近场说话人。
- 房间混响。

### 最小采集方案

即使没有完整实验条件，也建议采少量真实数据：

```text
安静近场说话：10-20 分钟
室内空调/电脑风扇：10 分钟
走廊/办公室环境：10 分钟
风噪/衣物摩擦：5-10 分钟
不同佩戴位置：每类 2-3 个位置
```

用途不是直接大规模训练，而是：

- 做真实 validation set。
- 发现合成数据没覆盖的噪声。
- 微调最后几层或 gate。
- 校准后处理和 loudness。

## 7. 后处理定位

后处理仍然有价值，但只能作为辅助：

- 防止输出过大。
- 控制高频 residual。
- 做轻量 loudness match。
- 对高 SNR 样本 bypass。

但它不能代替：

```text
更强前端 + 复数域建模 + 真实数据
```

## 8. 推荐下一步执行顺序

### Step 1：跑 oracle MWF 上限

先生成 oracle MWF demo，直接听它能否解决呼呼声。

如果 oracle MWF 仍然不行，先别训练新模型，转去做数据和评估集。

### Step 2：增加听感相关 metrics

给所有版本补：

```text
high_band_residual_ratio
speech_band_rms_ratio
quiet_frame_attenuation
```

用这些指标解释为什么某些 SI-SDR 高的版本仍然难听。

### Step 3：teacher-student

如果 oracle MWF 好听：

```text
oracle MWF / larger teacher -> generate teacher enhanced
tiny model -> learn teacher masks/filter
```

这比小模型直接学 clean 更稳。

### Step 4：真实双麦小验证集

采少量真实双麦数据，只要能覆盖“呼呼声”来源，就可以作为后续模型选择依据。

### Step 5：再考虑端侧压缩

等听感方向正确后，再做：

- INT8。
- Q15。
- C reference。
- CMSIS-DSP/CMSIS-NN。

不要过早把一个听感不好的模型端侧化。

## 9. 当前结论

后续真正值得做的是：

```text
oracle/teacher MWF 诊断
  -> 感知指标
  -> teacher-student 复数/多帧滤波
  -> 真实双麦验证集
```

不建议继续：

```text
调 dehiss 参数
调 TinyGRU hidden size
只看 SI-SDR
只做 band mask
```
