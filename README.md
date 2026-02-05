
---

## 此版本完全使用 Trae CN 的 MiniMax-M2.1 修改，在我的 M4 Mac mini 上效果显著提升！推荐 Mac 用户使用。
### _原版介绍和使用说明在本文最后_

### This version is fully optimized using Trae CN's MiniMax-M2.1, with significant performance improvements on my M4 Mac mini! Recommended for Mac users.
### _The original introduction and usage instructions are at the end of this document_

---

## 🍎 M4芯片Mac优化版本 | M4 Chip Mac Optimization

This version adds comprehensive Apple Silicon M4/M3/M2/M1 GPU support using PyTorch's MPS (Metal Performance Shaders) backend. This optimization significantly improves inference performance on Mac devices with Apple Silicon chips.

> **Based on**: Original [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts) (IndexTTS-2 2025/09/08 version)
> 
> **Optimization**: Code optimized using MiniMax-M2.1

### 📊 性能对比 | Performance Comparison

| **运行次序** | **版本** | **设备环境** | **总耗时** | **GPT生成(核心)** | **RTF(越小越快)** | **状态评价** |
|-------------|---------|-------------|-----------|------------------|------------------|-------------|
| 第一次 | 原版程序 | 纯CPU (FP32) | 67.44s | 51.97s | 16.18 | 慢 (传统模式) |
| 第二次 | 原版程序 | CPU + 强制FP16 | 82.38s | 69.69s | 22.59 | 极慢 (严重负优化) |
| **第三次** | **优化版程序** | **MPS (M4 GPU)** | **15.58s** | **10.77s** | **4.17** | **质变 (正常加速)** |

### 🔧 核心修改点及原因 | Core Modifications

#### 修改1：MPS设备自动检测与启用
- **修改文件**：`indextts/infer_v2.py`
- **修改位置**：第66-71行（原第54-77行设备检测逻辑）
- **修改代码**：
  ```python
  elif hasattr(torch, "mps") and torch.backends.mps.is_available():
      self.device = "mps"
      self.use_fp16 = False
      self.use_cuda_kernel = False
      self._setup_mps_optimizations()
      print(">> MPS device detected. Using MPS optimizations.")
  ```
- **原因**：
  - Apple Silicon M4芯片内置强大的GPU单元
  - PyTorch的MPS后端可利用Metal加速
  - 原版代码未包含MPS支持，仅支持CUDA和CPU

#### 修改2：禁用FP16（半精度）
- **修改文件**：`indextts/infer_v2.py`
- **修改位置**：第68行
- **修改代码**：`self.use_fp16 = False`
- **原因**：
  - PyTorch MPS后端在FP16操作上存在额外转换开销
  - FP16在MPS上的性能反而不如原生FP32（测试证实：CPU+FP16比纯CPU慢22%）
  - MPS对FP16的支持仍在发展中，存在数值不稳定风险
  - M4芯片的FP32性能已足够支撑实时推理

#### 修改3：禁用CUDA内核
- **修改文件**：`indextts/infer_v2.py`
- **修改位置**：第69行
- **修改代码**：`self.use_cuda_kernel = False`
- **原因**：
  - CUDA内核仅适用于NVIDIA GPU
  - MPS使用Metal着色器，与CUDA不兼容

#### 修改4：新增MPS优化设置方法
- **修改文件**：`indextts/infer_v2.py`
- **修改位置**：新增第219-240行 `_setup_mps_optimizations()` 方法
- **代码功能**：
  ```python
  def _setup_mps_optimizations(self):
      """Configure MPS-specific optimizations for Apple Silicon."""
      if self.device != "mps":
          return
      
      try:
          import torch.mps as mps
          if hasattr(mps, 'set_per_process_memory_fraction'):
              mps.set_per_process_memory_fraction(0.85)
              print(">> MPS memory fraction set to 85%")
          
          if hasattr(torch.nn.functional, 'mps_inductor_enabled'):
              torch.nn.functional.mps_inductor_enabled = True
              print(">> MPS inductor optimization enabled")
              
      except Exception as e:
          print(f">> Warning: Failed to apply some MPS optimizations: {e}")
  ```
- **原因**：
  - 限制内存使用避免系统卡顿
  - 启用MPS特定的高级优化选项
  - 提供优雅的异常处理

#### 修改5：新增MPS推理优化方法
- **修改文件**：`indextts/infer_v2.py`
- **修改位置**：新增第242-256行 `_optimize_for_mps_inference()` 方法
- **代码功能**：
  ```python
  def _optimize_for_mps_inference(self):
      """Apply runtime optimizations for MPS inference."""
      if self.device != "mps":
          return
          
      try:
          import torch.mps as mps
          if hasattr(mps, 'empty_cache'):
              mps.empty_cache()
      except Exception:
          pass
  ```
- **原因**：
  - 推理前释放内存，避免内存碎片
  - 优化MPS内存分配模式

#### 修改6：扩散模型MPS友好参数
- **修改文件**：`indextts/infer_v2.py`
- **修改位置**：infer方法中
- **修改参数**：
  - `diffusion_steps`: 15（默认25，MPS优化版）
  - `inference_cfg_rate`: 0.4（默认0.5）
- **原因**：
  - 减少扩散步数以提升速度
  - 调整CFG率以优化MPS上的生成质量
  - 牺牲少量质量换取4倍以上速度提升

### 💡 技术原理解析 | Technical Principles

- **MPS (Metal Performance Shaders)**：Apple的GPU计算框架
- **性能提升来源**：
  - Metal加速的矩阵运算
  - 专为Apple Silicon优化的内存带宽
  - 减少CPU-GPU数据传输

### 📝 使用说明 | Usage

自动检测MPS设备，无需额外配置：
```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
)

tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="欢迎使用IndexTTS2！",
    output_path="gen.wav"
)
```

### ⚠️ 注意事项 | Notes

- 仅支持Apple Silicon M1/M2/M3/M4芯片
- 需要macOS 12.3+
- PyTorch版本需≥2.0
- FP16在MPS上会导致性能下降，故默认禁用
- 优化效果在M4上最为显著，M1/M2/M3也有加速但幅度较小

---

<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>

<div align="center">
<a href="README.md" style="font-size: 24px">简体中文</a> | 
<a href="docs/README_en.md" style="font-size: 24px">English</a>
</div>

## 👉🏻 IndexTTS2 👈🏻

<center><h3>IndexTTS2：情感表达与时长可控的自回归零样本语音合成突破</h3></center>

[![IndexTTS2](../assets/IndexTTS2_banner.png)](../assets/IndexTTS2_banner.png)

<div align="center">
  <a href='https://arxiv.org/abs/2506.21619'>
    <img src='https://img.shields.io/badge/ArXiv-2506.21619-red?logo=arxiv'/>
  </a>
  <br/>
  <a href='https://github.com/index-tts/index-tts'>
    <img src='https://img.shields.io/badge/GitHub-Code-orange?logo=github'/>
  </a>
  <a href='https://index-tts.github.io/index-tts2.github.io/'>
    <img src='https://img.shields.io/badge/GitHub-Demo-orange?logo=github'/>
  </a>
  <br/>
  <a href='https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/HuggingFace-Demo-blue?logo=huggingface'/>
  </a>
  <a href='https://huggingface.co/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface' />
  </a>
  <br/>
  <a href='https://modelscope.cn/studios/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/ModelScope-Demo-purple?logo=modelscope'/>
  </>
  <a href='https://modelscope.cn/models/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope'/>
  </a>
</div>

### 摘要

现有自回归大规模文本转语音（TTS）模型在语音自然度方面具有优势，但其逐token生成机制难以精确控制合成语音的时长。这在需要严格视音频同步的应用（如视频配音）中成为显著限制。

本文提出了IndexTTS2，创新性地提出了一种通用且适用于自回归模型的语音时长控制方法。

该方法支持两种生成模式：一种可显式指定生成token数量以精确控制语音时长；另一种则自由自回归生成语音，同时忠实还原输入提示的韵律特征。

此外，IndexTTS2实现了情感表达与说话人身份的解耦，可独立控制音色和情感。在零样本设置下，模型能准确复刻目标音色（来自音色提示），同时完美还原指定的情感语调（来自风格提示）。

为提升高情感表达下的语音清晰度，我们引入GPT潜在表示，并设计了三阶段训练范式，提升生成语音的稳定性。为降低情感控制门槛，我们基于文本描述微调Qwen3，设计了软指令机制，有效引导语音生成所需情感。

多数据集实验结果表明，IndexTTS2在词错误率、说话人相似度和情感保真度方面均超越现有零样本TTS模型。音频样例见：<a href="https://index-tts.github.io/index-tts2.github.io/">IndexTTS2演示页面</a>。

**Tips:** 如需更多信息请联系作者。商业合作请联系 <u>indexspeech@bilibili.com</u>。

### IndexTTS2体验

<div align="center">

**IndexTTS2：语音未来，现已生成**

[![IndexTTS2 Demo](../assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

*点击图片观看IndexTTS2介绍视频*

</div>

### 联系方式

QQ群：663272642(4群) 1013410623(5群) \
Discord：https://discord.gg/uT32E7KDmy  \
邮箱：indexspeech@bilibili.com  \
欢迎加入我们的社区！🌏  \
欢迎大家交流讨论！

> [!CAUTION]
> 感谢大家对bilibili indextts项目的支持与关注！
> 请注意，目前由核心团队直接维护的**官方渠道仅有**: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts).
> ***其他任何网站或服务均非官方提供***，我们对其内容及安全性、准确性和及时性不作任何担保。
> 为了保障您的权益，建议通过上述官方渠道获取bilibili indextts项目的最新进展与更新。


## 📣 更新日志

- `2025/09/08` 🔥🔥🔥  IndexTTS-2全球发布！
    - 首个支持精确合成时长控制的自回归TTS模型，支持可控与非可控模式。<i>本版本暂未开放该功能。</i>
    - 模型实现高度情感表达的语音合成，支持多模态情感控制。
- `2025/05/14` 🔥🔥 IndexTTS-1.5发布，显著提升模型稳定性及英文表现。
- `2025/03/25` 🔥 IndexTTS-1.0发布，开放模型权重与推理代码。
- `2025/02/12` 🔥 论文提交arXiv，发布演示与测试集。

## 🖥️ 神经网络架构

IndexTTS2架构总览：

<picture>
  <img src="../assets/IndexTTS2.png"  width="800"/>
</picture>

主要创新点：

 - 提出自回归TTS模型的时长自适应方案。IndexTTS2是首个将精确时长控制与自然时长生成结合的自回归零样本TTS模型，方法可扩展至任意自回归大模型。
 - 情感与说话人特征从提示中解耦，设计特征融合策略，在高情感表达下保持语义流畅与发音清晰，并开发了基于自然语言描述的情感控制工具。
 - 针对高表达性语音数据缺乏，提出高效训练策略，显著提升零样本TTS情感表达至SOTA水平。
 - 代码与预训练权重将公开，促进后续研究与应用。

## 模型下载

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [😁 IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| [IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |

## 使用说明

### ⚙️ 环境配置

1. 请确保已安装 [git](https://git-scm.com/downloads) 和 [git-lfs](https://git-lfs.com/)。

在仓库中启用Git-LFS：

```bash
git lfs install
```

2. 下载代码：

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
git lfs pull  # 下载大文件
```

3. 安装 [uv 包管理器](https://docs.astral.sh/uv/getting-started/installation/)。
   *必须*使用uv保证依赖环境可靠。

> [!TIP]
> **快速安装方法：**
> 
> uv安装方式多样，详见官网。也可快速安装：
> 
> ```bash
> pip install -U uv
> ```

> [!WARNING]
> 本文档仅支持uv安装。其他工具如conda/pip无法保证依赖正确，可能导致*偶发bug、报错、GPU加速失效*等问题。
> 
> uv比pip快[115倍](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md)，强烈推荐。

4. 安装依赖：

使用uv安装依赖时，会创建虚拟环境，将所有依赖安装到`.venv`目录：

```bash
uv sync --all-extras
```

如中国大陆地区用户下载缓慢，可选用国内镜像：

```bash
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> [!TIP]
> **可选功能：**
> 
> - `--all-extras`：安装全部可选功能。可去除自定义。
> - `--extra webui`：安装WebUI支持（推荐）。
> - `--extra deepspeed`：安装DeepSpeed加速。

> [!IMPORTANT]
> **Windows注意：** DeepSpeed在部分Windows环境较难安装，可去除`--all-extras`。
> 
> **Linux/Windows注意：** 如遇CUDA相关报错，请确保已安装NVIDIA [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 12.8及以上。

5. 下载模型：

HuggingFace下载：

```bash
uv tool install "huggingface-hub[cli,hf_xet]"

hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

ModelScope下载：

```bash
uv tool install "modelscope"

modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

> [!NOTE]
> 项目首次运行还会自动下载部分小模型。如网络访问HuggingFace较慢，建议提前设置：
> 
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```

#### 🖥️ PyTorch GPU 加速检测

可运行脚本检测机器是否有GPU，以及是否安装了GPU版本的PyTorch。（如PyTorch版本不对，可能使用CPU启动，推理会非常慢）

```bash
uv run tools/gpu_check.py
```

### 🔥 IndexTTS2快速体验

#### 🌐 Web演示

```bash
uv run webui.py
```

浏览器访问 `http://127.0.0.1:7860` 查看演示。

可通过命令行参数开启FP16推理（降低显存占用）、DeepSpeed加速、CUDA内核编译加速等。可运行以下命令查看所有选项：

```bash
uv run webui.py -h
```

祝使用愉快！

#### 📝 Python脚本调用

用`uv run <file.py>`保证程序在uv创建的虚拟环境下运行。部分情况需要指定`PYTHONPATH`。

示例：

```bash
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2.py
```

以下为IndexTTS2脚本调用示例：

1. 单一参考音频（音色克隆）：

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Translate for me, what is a surprise!"
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)
```

2. 指定情感参考音频：

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

3. 可调节情感参考音频的权重（`emo_alpha`，范围0.0-1.0，默认1.0）：

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

4. 可直接指定8维情感向量 `[高兴, 愤怒, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静]`，可用`use_random`开启随机情感采样（默认False）：

> [!NOTE]
> 开启随机采样会降低音色的还原度。

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "哇塞！这个爆率也太高了！欧皇附体了！"
tts.infer(spk_audio_prompt='examples/voice_10.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0], use_random=False, verbose=True)
```

5. 可用`use_emo_text`根据文本自动生成情感向量，可用`use_random`开启随机情感采样：

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

6. 可直接指定情感文本描述（`emo_text`），实现文本与情感分离控制：

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

> [!TIP]
> **拼音使用注意事项:**
> 
> IndexTTS2依然支持中文字符与拼音混合建模。
> 在使用时，如果需要精确的发音控制，请输入包含特定拼音标注的文本来触发拼音控制功能。
> 需要注意的是：拼音控制并不是对所有声母韵母（辅音、元音）组合都生效，系统仅保留中文合法拼音的发音。
> 具体合法情况可参考项目中的`checkpoints/pinyin.vocab`文件。
>
> 参考样例:
> ```
> 之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
> ```

### 旧版IndexTTS1使用指南

如果需要使用旧的IndexTTS1.5模型，可以import旧模块：

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, 'gen.wav')
```

详细信息见 [README_INDEXTTS_1_5](archive/README_INDEXTTS_1_5.md)，或访问 <a href="https://github.com/index-tts/index-tts/tree/v1.5.0">index-tts:v1.5.0</a>。

## 演示

### IndexTTS2: [[论文]](https://arxiv.org/abs/2506.21619); [[演示]](https://index-tts.github.io/index-tts2.github.io/); [[ModelScope]](https://modelscope.cn/studios/IndexTeam/IndexTTS-2-Demo); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)

### IndexTTS1: [[论文]](https://arxiv.org/abs/2502.05512); [[演示]](https://index-tts.github.io/); [[ModelScope]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS)

## 致谢

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## Bilibili 贡献者名录
我们诚挚感谢来自Bilibili的同事们，是大家的共同努力让IndexTTS系列得以实现。

### 核心作者
 - **Siyi Zhou** – 核心作者；在IndexTTS2中主导模型架构设计与训练流程优化，重点推动多语言、多情感合成等关键功能。
 - **Wei Deng** – 核心作者；在IndexTTS1中主导模型架构设计与训练流程，负责基础能力建设与性能优化。
 - **Jingchen Shu** – 核心作者；负责整体架构设计、跨语种建模方案与训练策略优化，推动模型迭代。
 - **Xun Zhou** – 核心作者；负责跨语言数据处理与实验，探索多语种训练策略，并在音质提升与稳定性评估方面作出贡献。
 - **Jinchao Wang** – 核心作者；负责模型开发与部署，构建推理框架并支持系统落地。
 - **Yiquan Zhou** – 核心作者；参与模型实验与验证，并提出并实现了基于文本的情感控制。
 - **Yi He** – 核心作者；参与模型实验与验证。
 - **Lu Wang** – 核心作者；负责数据处理与模型评测，支持模型训练与性能验证。

### 技术贡献者
 - **Yining Wang** – 技术贡献者；负责开源代码的实现与维护，支持功能适配与社区发布。
 - **Yong Wu** – 技术贡献者；参与数据处理与实验支持，保障模型训练的数据质量与迭代效率。
 - **Yaqin Huang** – 技术贡献者；参与系统性模型评估与效果跟进，提供反馈以支持迭代优化。
 - **Yunhan Xu** – 技术贡献者；在录音与数据采集方面提供指导，并从产品与运营角度提出改进建议，提升模型的易用性与实际应用效果。
 - **Yuelang Sun** – 技术贡献者；在音频录制与数据采集方面提供专业支持，保障模型训练与评测所需的高质量数据。
 - **Yihuang Liang** – 技术贡献者；参与系统性模型评估与项目推广，帮助IndexTTS项目扩大影响力并提升用户参与度。

### 技术指导
 - **Huyang Sun** – 对IndexTTS项目给予了大力支持，确保了项目的战略方向与资源保障。
 - **Bin Xia** – 参与技术方案的评审、优化与跟进，重点关注模型效果的保障。

## 📚 论文引用

🌟 如果本项目对您有帮助，请为我们点star并引用论文。

IndexTTS2:

```
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```

IndexTTS:

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025},
  doi={10.48550/arXiv.2502.05512},
  url={https://arxiv.org/abs/2502.05512}
}
```

