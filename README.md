# TinyHear Dual-Mic Denoising

[![CI](https://github.com/turambar928/TinyHear-Dual-Mic-Denoising/actions/workflows/ci.yml/badge.svg)](https://github.com/turambar928/TinyHear-Dual-Mic-Denoising/actions/workflows/ci.yml)

双麦端侧上行降噪原型。

这个项目实现一个面向助听器/耳戴设备的双麦上行降噪原型：双通道音频输入，模型预测 32 个频带的软掩蔽，最后对参考麦克风频谱做增强重构。

## Demo

- 本地网页试听 demo：`http://127.0.0.1:38179/runs/audio_demo/index.html`
- 当前 demo 包含 `c120_h32_gate`、`c120_h2_gate`、`c116_h24_gate` 三组 noisy/clean/enhanced 对比。
- 如果本地服务没启动，可在项目根目录运行 `python3 -m http.server 38179`，然后打开上面的链接。

## 当前结果

- 推荐基线：CMU ARCTIC clean speech + DEMAND 多通道环境噪声。
- 推荐部署方案：`c116` 空间特征 TCN + `hidden=24` learned gate。
- 模型规模：TCN `140,276` 参数，gate `9,265` 参数，总计 `149,541` 参数，满足 150K 目标。
- 实时链路：16 kHz，256 点 FFT，64 samples hop，4 ms 步进。
- Python eval：全量 SI-SDR improvement `4.474 dB`，high-SNR 退化率 `0%`。
- C Q15 模型 reference：mean abs diff `0.01703`，streaming 与 batch Q15 完全一致。
- C learned gate reference：gate abs diff `0.0000039` against Python gate。
- C gated realtime DSP reference：mean abs diff `0.00078` against Python gated realtime reference。
- PC reference benchmark：Q15 模型约 `0.322 ms/frame`，完整 gated C realtime reference 约 `2.582 ms/hop`。
- C 端状态内存：`TinyTcnState` 13.9 KB，`TinyGateState` 1.5 KB，`TinyRealtimeDspState` 19.6 KB。

## 方案摘要

- 采样率：16 kHz。
- 分帧：256 点 FFT，64 点 hop，算法步进 4 ms；模型严格因果，不使用未来帧。
- 输入特征：参考麦与副麦的 32 频带 log power、频带能量比、IPD cos/sin、coherence，共 192 维。
- 模型：Causal depthwise-separable TCN，推荐 `channels=116, blocks=8`。
- learned gate：对整段/前缀特征做 mean/std pooling，MLP 判断增强或旁路，减少高 SNR 过处理。
- 参数量：TCN + gate 共约 149.5K 参数，满足 100-150 KB 目标。
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

# 下一轮推荐训练：IPD/coherence 空间特征 + 150KB 内强模型 + 谱幅度 loss
python scripts/train.py --data data/synth --epochs 20 --batch-size 8 --out runs/tiny_tcn_spatial --spatial-features --channels 120 --band-mag-loss-weight 0.1

# 导出 INT8 权重和 C 头文件
python scripts/export_int8.py --checkpoint runs/tiny_tcn/best.pt --out runs/tiny_tcn/int8

# 离线增强一个双通道 wav
python scripts/enhance_wav.py --checkpoint runs/tiny_tcn/best.pt --input data/synth/val/mix_0000.wav --output enhanced.wav

# 逐帧流式模型增强一个双通道 wav
python scripts/enhance_streaming.py --checkpoint runs/tiny_tcn/best.pt --input data/synth/val/mix_0000.wav --output enhanced_streaming.wav

# 完整实时链路增强：流式 STFT + 逐帧模型 + IRFFT overlap-add
python scripts/enhance_realtime.py --checkpoint runs/tiny_tcn/best.pt --input data/synth/val/mix_0000.wav --output enhanced_realtime.wav

# 评估预生成验证集
python scripts/evaluate.py --checkpoint runs/tiny_tcn/best.pt --data data/synth --split val --save-audio runs/tiny_tcn/eval_audio

# 对比离线模型和逐帧流式模型输出
python scripts/compare_streaming.py --checkpoint runs/tiny_tcn/best.pt --data data/synth --split val

# 对比离线增强和完整实时链路输出
python scripts/compare_realtime.py --checkpoint runs/tiny_tcn/best.pt --data data/synth --split val

# 开启 high-SNR bypass，减少干净场景过处理
python scripts/compare_realtime.py --checkpoint runs/tiny_tcn/best.pt --data data/synth --split val --high-snr-bypass

# 生成 noisy/clean/offline/realtime 听感对比样例
python scripts/make_listening_eval.py --checkpoint runs/tiny_tcn/best.pt --data data/synth --split val --out runs/tiny_tcn/listening_eval

# 可选感知指标，安装 pystoi/pesq 后会额外输出 STOI/PESQ
python scripts/evaluate_perceptual.py --clean-dir runs/tiny_tcn/listening_eval --enhanced-dir runs/tiny_tcn/listening_eval --pattern "sample_*_realtime.wav"

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

当前推荐 `c116 + h24 gate` 端侧链路复现命令：

```bash
PYTHONPATH=src python3 scripts/export_int8.py \
  --checkpoint runs/arctic_demand_spatial_c116/best.pt \
  --out runs/arctic_demand_spatial_c116/int8

PYTHONPATH=src python3 scripts/calibrate_int8.py \
  --checkpoint runs/arctic_demand_spatial_c116/best.pt \
  --export-dir runs/arctic_demand_spatial_c116/int8 \
  --data data/arctic_demand_eval \
  --split val \
  --max-items 160 \
  --percentile 100

PYTHONPATH=src:scripts python3 scripts/export_gate.py \
  --gate runs/gate_spatial_c116_h24/best.pt \
  --out c_reference/generated/tiny_gate_params.h

PYTHONPATH=src:scripts python3 scripts/dump_c_reference_assets.py \
  --checkpoint runs/arctic_demand_spatial_c116/best.pt \
  --export-dir runs/arctic_demand_spatial_c116/int8 \
  --input-wav data/arctic_demand_eval/val/mix_0000.wav \
  --gate runs/gate_spatial_c116_h24/best.pt \
  --out-dir c_reference/generated \
  --frames 16

make -C c_reference clean run
make -C c_reference clean bench
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
- `docs/performance.md`：PC 侧 C reference benchmark 和内存估算。
- `docs/cmsis_porting.md`：CMSIS-DSP/CMSIS-NN 上板替换说明。
- `docs/roadmap.md`：后续工程和研究任务清单。
- `docs/data_upgrade.md`：更强公开数据和真实双麦数据升级路线。
- `docs/effect_improvement.md`：听感改进项、bypass、强模型和感知指标说明。
- `docs/project_report.md`：项目报告，可用于实习总结/答辩材料。
- `docs/slides_outline.md`：汇报 PPT 提纲。
- `docs/listening_eval.md`：听感样例生成方式、样例列表和指标。
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
- `scripts/enhance_streaming.py`：逐帧模型状态增强，模拟端侧连续帧推理。
- `scripts/compare_streaming.py`：对比离线模型与逐帧流式模型的 mask、waveform 和 SI-SDR 差异。
- `scripts/enhance_realtime.py`：完整实时链路增强，包含 causal STFT、模型状态、IRFFT 和 overlap-add。
- `scripts/compare_realtime.py`：评估完整实时链路相对离线路径的延迟、SI-SDR 和 waveform 差异。
- `scripts/make_listening_eval.py`：生成 noisy/clean/offline/realtime 听感对比样例。
- `scripts/evaluate_perceptual.py`：可选 STOI/PESQ 感知指标评估入口。
- `scripts/evaluate.py`：SI-SDR improvement 和 mask MSE 评估。
- `scripts/materialize_mixes.py`：将 on-the-fly clean/noise 数据固化为可复现 mix/clean 样本。
- `scripts/dump_c_reference_assets.py`：生成 C reference 所需 scale 头文件和测试向量。
- `scripts/export_gate.py`：导出 learned gate 的 C header 参数。
- `c_reference/`：PC 侧 C INT8 + realtime DSP reference，对齐 Python fixed-scale/realtime reference。

C reference 现在包含两条路径：

- `tiny_tcn_forward`：int8 卷积/int32 累加，scale 用 float，便于和 Python reference 对齐。
- `tiny_tcn_forward_q15`：中间激活 int8、requant multiplier/shift、最终 mask 为 Q15，核心推理全整型。
- `tiny_tcn_process_frame_q15`：逐帧流式接口，内部缓存每个 TCN block 的 depthwise 历史状态，适合端侧 4 ms hop 连续运行。
- `tiny_realtime_process_hop`：完整 C 端实时 reference，包含 causal STFT 特征、Q15 模型、mask-to-bin、IRFFT 和 overlap-add。
- `fft_backend.h/.c`：RFFT/IRFFT 抽象层；PC reference 默认 naive 实现，端侧可替换为 CMSIS-DSP。
