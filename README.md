# TinyHear Dual-Mic Denoising

[![CI](https://github.com/turambar928/TinyHear-Dual-Mic-Denoising/actions/workflows/ci.yml/badge.svg)](https://github.com/turambar928/TinyHear-Dual-Mic-Denoising/actions/workflows/ci.yml)

双麦端侧上行降噪原型。

这个项目实现一个面向助听器/耳戴设备的双麦上行降噪原型：双通道音频输入，模型预测 32 个频带的软掩蔽，最后对参考麦克风频谱做增强重构。

## 方案摘要

- 采样率：16 kHz。
- 分帧：256 点 FFT，64 点 hop，算法步进 4 ms；模型严格因果，不使用未来帧。
- 输入特征：参考麦与副麦的 32 频带 log power、频带能量比，共 96 维。
- 模型：Causal depthwise-separable TCN，默认 `channels=112, blocks=8`。
- 参数量：约 119K 参数，INT8 权重约 119 KB，满足 100-150 KB 目标。
- 输出：32 频带 mask，插值到 FFT bin 后乘到参考麦复数谱。
- 整型路径：`scripts/export_int8.py` 会导出 per-tensor INT8 权重、scale 和 C 头文件；端侧可用 int8 卷积 + int32 累加实现。

## 数据集建议

优先级从易落地到更真实：

1. Microsoft DNS Challenge：直接提供语音、噪声、RIR 合成脚本，适合噪声抑制训练。
2. LibriSpeech 或 Mini LibriSpeech + MUSAN：语音和噪声分别公开，适合快速原型。
3. 后续真实双麦：录制目标说话人在 0 度方向、干扰噪声多方向的设备阵列数据，用来微调空间特征。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 生成一小批合成双麦训练样本
python scripts/make_synth_dataset.py --out data/synth --seconds 2 --num-train 200 --num-val 20

# 训练
python scripts/train.py --data data/synth --epochs 10 --batch-size 8 --out runs/tiny_tcn

# 导出 INT8 权重和 C 头文件
python scripts/export_int8.py --checkpoint runs/tiny_tcn/best.pt --out runs/tiny_tcn/int8

# 离线增强一个双通道 wav
python scripts/enhance_wav.py --checkpoint runs/tiny_tcn/best.pt --input data/synth/val/mix_0000.wav --output enhanced.wav

# 评估预生成验证集
python scripts/evaluate.py --checkpoint runs/tiny_tcn/best.pt --data data/synth --split val --save-audio runs/tiny_tcn/eval_audio

# 验证导出的 INT8 权重 reference
python scripts/verify_int8_reference.py --checkpoint runs/tiny_tcn/best.pt --export-dir runs/tiny_tcn/int8

# 用固定验证集校准 activation scale，并按固定 scale 验证
python scripts/calibrate_int8.py --checkpoint runs/tiny_tcn/best.pt --export-dir runs/tiny_tcn/int8 --data data/synth --split val
python scripts/verify_int8_reference.py --checkpoint runs/tiny_tcn/best.pt --export-dir runs/tiny_tcn/int8 --fixed-scales

# 生成并运行 C reference 测试向量
python scripts/dump_c_reference_assets.py \
  --checkpoint runs/yesno_public/best.pt \
  --export-dir runs/yesno_public/int8 \
  --input-wav data/yesno_public_eval/val/mix_0000.wav \
  --out-dir c_reference/generated \
  --frames 16
make -C c_reference run
```

公开数据集下载后也可以直接训练，只要按下面结构放置 wav：

```text
data/custom/
  train/
    clean/*.wav
    noise/*.wav
  val/
    clean/*.wav
    noise/*.wav
```

然后运行：

```bash
python scripts/train.py --data data/custom --on-the-fly
```

如果原始语音和噪声在不同目录，可以先抽样整理：

```bash
PYTHONPATH=src python scripts/prepare_wav_dataset.py \
  --clean-root /path/to/clean_speech_wavs \
  --noise-root /path/to/noise_wavs \
  --rir-root /path/to/optional_rir_wavs \
  --out data/public_small \
  --train-clean 5000 --train-noise 1000 --val-clean 200 --val-noise 100

PYTHONPATH=src python scripts/train.py --data data/public_small --on-the-fly --epochs 30 --batch-size 16 --out runs/public_small
```

`rir` 目录是可选的。RIR wav 如果是双通道，会被当成双麦房间响应；如果是单通道，会与 TDOA/衰减模拟组合使用。

更适配双麦训练的公开数据组合是 **LibriSpeech/CMU ARCTIC clean speech + DEMAND 多通道环境噪声**：

```bash
# 推荐路线 A：LibriSpeech/Mini LibriSpeech FLAC -> clean wav
PYTHONPATH=src python scripts/prepare_librispeech_wavs.py \
  --src downloads/mini_librispeech \
  --out data/libri_demand \
  --train-count 500 --val-count 80

# 推荐路线 B：Hugging Face LibriSpeech streaming -> clean wav
PYTHONPATH=src python scripts/prepare_hf_librispeech.py \
  --out data/libri_demand \
  --train-count 500 --val-count 80

# DEMAND 多通道噪声 -> stereo noise wav
PYTHONPATH=src python scripts/prepare_demand_noise.py \
  --src downloads/demand \
  --out data/libri_demand \
  --train-count 500 --val-count 100

PYTHONPATH=src python scripts/train.py \
  --data data/libri_demand \
  --on-the-fly \
  --seconds 2 \
  --epochs 30 \
  --batch-size 8 \
  --out runs/libri_demand
```

如果 LibriSpeech 下载受限，可用 CMU ARCTIC 作为更小但比 YESNO 丰富得多的 clean speech baseline：

```bash
PYTHONPATH=src python scripts/prepare_wav_dataset.py \
  --clean-root downloads/cmu_arctic/extracted \
  --noise-root data/yesno_demand/train/noise \
  --out data/arctic_demand \
  --train-clean 800 --train-noise 1 --val-clean 160 --val-noise 1

PYTHONPATH=src python scripts/prepare_demand_noise.py \
  --src downloads/demand \
  --out data/arctic_demand \
  --train-count 800 --val-count 160

PYTHONPATH=src python scripts/train.py \
  --data data/arctic_demand \
  --on-the-fly \
  --seconds 2 \
  --epochs 40 \
  --batch-size 8 \
  --out runs/arctic_demand
```

已经有 checkpoint 时可继续训练：

```bash
PYTHONPATH=src python scripts/train.py \
  --data data/arctic_demand \
  --on-the-fly \
  --seconds 2 \
  --epochs 20 \
  --batch-size 8 \
  --lr 3e-4 \
  --out runs/arctic_demand \
  --resume runs/arctic_demand/best.pt \
  --start-epoch 20
```

`--on-the-fly` 训练集如果要固定评估样本，可以先 materialize：

```bash
PYTHONPATH=src python scripts/materialize_mixes.py --data data/public_small --split val --out data/public_small_eval --count 100
PYTHONPATH=src python scripts/evaluate.py --checkpoint runs/public_small/best.pt --data data/public_small_eval --split val
```

## 目录

- `docs/implementation_plan.md`：完整实现方案、端侧映射和下一步路线。
- `docs/experiments.md`：训练数据选择、实验记录和指标。
- `src/ha_denoise/model.py`：100-150KB 目标模型。
- `src/ha_denoise/features.py`：双麦 STFT 特征、mask 标签、重构。
- `src/ha_denoise/dataset.py`：合成双麦数据与训练数据集。
- `scripts/train.py`：训练入口。
- `scripts/export_int8.py`：INT8 权重导出。
- `scripts/verify_int8_reference.py`：INT8 权重 + int32 卷积累加 reference。
- `scripts/calibrate_int8.py`：统计每层输入 activation scale 并写入 `model_int8.json`。
- `scripts/prepare_librispeech_wavs.py`：将 LibriSpeech/Mini LibriSpeech FLAC 转成 clean wav。
- `scripts/prepare_hf_librispeech.py`：从 Hugging Face 数据集导出 LibriSpeech clean wav。
- `scripts/prepare_demand_noise.py`：将 DEMAND 多通道噪声整理成双通道 noise wav。
- `scripts/enhance_wav.py`：离线增强。
- `scripts/evaluate.py`：SI-SDR improvement 和 mask MSE 评估。
- `scripts/materialize_mixes.py`：将 on-the-fly clean/noise 数据固化为可复现 mix/clean 样本。
- `scripts/dump_c_reference_assets.py`：生成 C reference 所需 scale 头文件和测试向量。
- `c_reference/`：PC 侧 C INT8 reference，对齐 Python fixed-scale reference。

C reference 现在包含两条路径：

- `tiny_tcn_forward`：int8 卷积/int32 累加，scale 用 float，便于和 Python reference 对齐。
- `tiny_tcn_forward_q15`：中间激活 int8、requant multiplier/shift、最终 mask 为 Q15，核心推理全整型。
- `tiny_tcn_process_frame_q15`：逐帧流式接口，内部缓存每个 TCN block 的 depthwise 历史状态，适合端侧 4 ms hop 连续运行。
