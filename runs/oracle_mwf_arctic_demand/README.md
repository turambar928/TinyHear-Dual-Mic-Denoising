# Oracle MWF Teacher 版本说明

## 1. 版本定位

`oracle_mwf_teacher` 是一个用于分析和生成 teacher target 的上限版本，不是最终可部署模型。

它和之前的 Tiny TCN、DeepFilter、TinyGRU、complex mask 模型不同：这个版本在评估时使用了已知的 `clean_*.wav` 来估计局部多通道 Wiener 滤波器。因此它可以回答一个关键问题：

```text
如果双麦前端足够强，当前数据上的噪声是否可以被明显压下去？
```

当前结果说明答案是肯定的。

固定 160 条 ARCTIC + DEMAND 验证集：

```text
mean_noisy_si_sdr:      4.33 dB
mean_enhanced_si_sdr:   20.39 dB
SI-SDR improvement:     +16.06 dB
```

这明显高于当前所有小模型版本：

```text
tiny_complex_mask:                +9.59 dB
tiny_deepfilter_coherence_mwf:    +9.46 dB
stable_postfilter:                +9.24 dB
tiny_gru_h136:                    +4.77 dB
```

所以这一版的意义不是直接部署，而是证明：

```text
强双麦前端 / MWF teacher 是有效方向。
后续小模型应该学习接近 MWF teacher 的滤波行为。
```

## 2. 输入输出

输入：

```text
mix_*.wav    双通道麦克风混合语音
clean_*.wav  双通道 clean reference
```

输出：

```text
oracle_mwf_*.wav  估计出的 clean mic0 语音
```

网页 demo 中每条样例包含：

```text
Noisy      原始 mic0
Offline    oracle MWF 输出
Realtime   同 Offline，目前用于网页对比
Clean      clean mic0 reference
```

## 3. 核心算法流程

### 3.1 STFT

先对双麦 mixture 和 clean reference 做 STFT：

```text
X0[f, t] = STFT(mix mic0)
X1[f, t] = STFT(mix mic1)
S0[f, t] = STFT(clean mic0)
S1[f, t] = STFT(clean mic1)
```

当前配置：

```text
sample_rate = 16000
n_fft       = 256
hop_length  = 64
window      = Hann
```

### 3.2 构造双麦观测向量

每个频点和时间帧有一个双通道观测：

```text
x[f, t] = [X0[f, t], X1[f, t]]^T
```

目标是估计 clean mic0：

```text
s[f, t] = S0[f, t]
```

### 3.3 局部统计估计

对每个频点，在时间维上做局部平均，估计：

```text
Rxx[f, t] = E[x x^H]
pxs[f, t] = E[x s^*]
```

其中：

```text
Rxx 是 2x2 mixture covariance
pxs 是 2x1 mixture-clean cross covariance
```

当前使用：

```text
window = 9 frames
diagonal_loading = 0.01
```

`window=9` 表示用 9 个 STFT 帧做局部统计，约等于：

```text
9 * 64 / 16000 = 36 ms
```

默认是 centered window，因此它是 oracle/teacher 上限，不是严格实时因果版本。

### 3.4 解局部 Wiener 滤波器

多通道 Wiener 滤波器形式：

```text
w[f, t] = Rxx[f, t]^-1 pxs[f, t]
```

双麦情况下，`Rxx` 是 2x2 矩阵，可以直接解析求逆：

```text
Rxx = [[a, b],
       [c, d]]

det = a*d - b*c

w0 = (d*p0 - b*p1) / det
w1 = (-c*p0 + a*p1) / det
```

为了避免病态矩阵，加入 diagonal loading：

```text
Rxx += diagonal_loading * trace(Rxx) * I
```

### 3.5 滤波输出

对每个频点和时间帧：

```text
Y[f, t] = w[f, t]^H x[f, t]
```

也就是：

```text
Y = conj(w0) * X0 + conj(w1) * X1
```

最后 ISTFT 得到增强波形：

```text
y[n] = ISTFT(Y)
```

## 4. 为什么这一版明显更好

之前的很多版本主要是：

```text
模型预测 gain 或 mask
然后乘到 noisy/beamformed spectrum
```

这种方法有几个限制：

- 只改幅度，不充分处理相位。
- band mask 频率分辨率太粗。
- 单通道/单帧处理难以区分 speech 和 residual noise。
- 后处理只能压噪声，不能真正重建 clean。

Oracle MWF 不同，它直接使用双麦观测和 clean reference 估计局部最优线性滤波器：

```text
双通道空间信息 + 局部时间统计 + 复数滤波
```

因此它能同时利用：

- 双麦空间差异。
- 复数相位信息。
- 局部多帧统计。
- clean target 的最优线性估计。

这解释了为什么它比单纯 mask / postfilter / TinyGRU 明显更强。

## 5. 为什么不能直接部署

这个版本使用了 clean reference：

```text
pxs = E[x s^*]
```

实际设备上没有 clean speech，因此不能直接部署。

它的作用是 teacher / upper bound：

```text
先证明强 MWF 路线可行
再让小模型学习估计接近这个 teacher 的滤波器或 mask
```

## 6. 后续可部署方案

推荐下一步做 teacher-student。

### 6.1 Teacher 生成

用 oracle MWF 离线生成 teacher enhanced：

```bash
PYTHONPATH=src python3 scripts/evaluate_oracle_mwf.py \
  --data data/arctic_demand_eval \
  --split val \
  --max-items 160 \
  --save-audio runs/oracle_mwf_arctic_demand/eval_audio \
  --save-listening runs/oracle_mwf_arctic_demand/listening_eval \
  --listening-samples 20 \
  --window 9 \
  --diagonal-loading 0.01
```

### 6.2 Student 模型目标

student 不直接使用 clean，而是从 mixture 特征估计：

```text
speech mask
noise mask
或 MWF filter coefficients
```

可选输出形式：

```text
方案 A：估计 speech/noise mask，DSP 端计算 2x2 MWF
方案 B：直接估计 w0/w1 复数滤波器
方案 C：估计 teacher complex spectrum / complex mask
```

在端侧最推荐方案 A：

```text
小模型估计 mask
端侧平滑 covariance
2x2 矩阵求逆
执行 MWF
```

原因是：

- 模型参数更少。
- DSP 更可控。
- 更符合 hearing-aid 常见 beamforming/MWF 结构。
- 比直接输出 129-bin complex mask 更稳定。

## 7. 当前文件结构

```text
runs/oracle_mwf_arctic_demand/
  README.md
  eval_audio/
    metrics.json
    oracle_mwf_*.wav       # 完整 160 条输出，默认不进入 git
  listening_eval/
    index.json
    sample_000_*.wav
    ...
    sample_019_*.wav       # 20 条网页试听样例
```

网页入口：

```text
runs/audio_demo/index.html
```

当前 demo 中 `oracle_mwf_teacher` 排在第一组。

## 8. 结论

`oracle_mwf_teacher` 是目前最重要的诊断版本。它证明：

```text
当前数据上的噪声可以被强双麦 MWF 明显压下去。
呼呼声问题不是完全无解。
继续调后处理不是主方向。
下一步应训练小模型去逼近 MWF teacher。
```

因此后续优化路线应从：

```text
调 gain / postfilter
```

转向：

```text
teacher-student MWF
复数域滤波
真实双麦验证集
```
