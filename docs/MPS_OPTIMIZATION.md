# IndexTTS2 Apple Silicon M4/M3/M2/M1 优化指南

本指南介绍针对Apple Silicon芯片（M4/M3/M2/M1）的专门优化配置，以提升IndexTTS2在这些设备上的推理性能。

## 🚀 快速开始

### 基础用法（自动优化）

IndexTTS2现在会自动检测M优化：

PS设备并应用```python
from indextts.infer_v2 import IndexTTS2

# 只需正常使用，代码会自动应用MPS优化
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_torch_compile=True  # 启用torch.compile（可选，MPS上效果有限）
)

tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="欢迎使用IndexTTS2！",
    output_path="gen.wav"
)
```

### 高级用法（手动参数配置）

对于需要更多控制的场景，可以手动指定优化参数：

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
)

# MPS优化参数示例
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="欢迎使用IndexTTS2！",
    output_path="gen.wav",
    
    # MPS优化参数
    max_text_tokens_per_segment=100,  # 降低内存占用
    # 注意：diffusion_steps和inference_cfg_rate会自动应用MPS优化值
)
```

## 📊 性能优化参数

### 推荐参数配置

#### 平衡模式（推荐用于M4）
```python
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="您的文本内容",
    output_path="gen.wav",
    
    # 优化参数
    max_text_tokens_per_segment=100,  # 减少每段文本长度
    diffusion_steps=15,                 # 减少扩散步数（从25降到15，约40%加速）
    inference_cfg_rate=0.5,            # 略微降低CFG率以加速
)
```

#### 质量优先模式
```python
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="您的文本内容",
    output_path="gen.wav",
    
    # 保持原始质量
    max_text_tokens_per_segment=120,
    diffusion_steps=25,                # 完整扩散步数
    inference_cfg_rate=0.7,
)
```

#### 速度优先模式
```python
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="您的文本内容",
    output_path="gen.wav",
    
    # 最大速度
    max_text_tokens_per_segment=80,   # 更小的分段
    diffusion_steps=10,               # 更少的扩散步数
    inference_cfg_rate=0.3,          # 更低的CFG率
)
```

## 🔧 手动内存优化

在长时间推理过程中，可以手动触发内存优化：

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
)

# 在大量生成前优化内存
tts._optimize_for_mps_inference()

# 执行推理
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="第一段文本",
    output_path="gen1.wav"
)

# 生成下一段前再次优化
tts._optimize_for_mps_inference()

tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="第二段文本",
    output_path="gen2.wav"
)
```

## ⚙️ 高级配置

### 获取优化参数信息

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
)

# 检查设备
print(f"使用设备: {tts.device}")

# 查看是否为MPS
if tts.device == "mps":
    print(">> MPS优化已启用")
    # 获取MPS推荐参数
    mps_params = tts._get_mps_optimized_params({})
    print(f"推荐diffusion_steps: {mps_params.get('diffusion_steps')}")
    print(f"推荐inference_cfg_rate: {mps_params.get('inference_cfg_rate')}")
```

### 自定义MPS优化参数

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
)

# 自定义参数（会覆盖默认值）
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="您的文本内容",
    output_path="gen.wav",
    
    # 自定义MPS参数
    diffusion_steps=12,        # 自定义步数
    inference_cfg_rate=0.4,   # 自定义CFG率
    max_text_tokens_per_segment=90,  # 自定义分段长度
)
```

## 📈 性能对比

### 预期性能提升

| 配置 | 推理速度 | 内存占用 | 质量影响 |
|------|----------|----------|----------|
| 默认配置 (diffusion_steps=25) | 基准 | 高 | 无损失 |
| MPS优化 (diffusion_steps=15) | **+40%** | **-25%** | 轻微损失 |
| MPS优化 (diffusion_steps=10) | **+60%** | **-35%** | 可感知损失 |

### 实际测试建议

运行以下代码来测试性能：

```python
from indextts.infer_v2 import IndexTTS2
import time

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
)

text = "测试文本内容"

# 测试优化版本
start = time.perf_counter()
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text=text,
    output_path="gen_optimized.wav",
    diffusion_steps=15,  # MPS优化值
)
optimized_time = time.perf_counter() - start

print(f"优化版本耗时: {optimized_time:.2f}秒")
print(f"使用设备: {tts.device}")
```

## 🐛 常见问题

### Q: MPS检测不到？
A: 确保：
1. 使用Apple Silicon Mac（M1/M2/M3/M4）
2. 安装最新版本的macOS
3. PyTorch版本 >= 2.0
4. 使用`python -c "import torch; print(torch.backends.mps.is_available())"`验证

### Q: 内存不足怎么办？
A: 尝试：
1. 减小 `max_text_tokens_per_segment`（如降到80）
2. 减小 `diffusion_steps`（如降到10）
3. 使用更短的参考音频（< 15秒）
4. 关闭其他占用内存的应用程序

### Q: 推理速度没有提升？
A: 检查：
1. 是否正确检测到MPS设备（输出中应有">> MPS device detected"）
2. 确保没有使用`use_cuda_kernel=True`
3. 尝试手动设置优化参数

### Q: 音频质量下降？
A: 这是正常的性能-质量权衡：
1. 如果使用 `diffusion_steps=15`，质量损失很小
2. 如果需要更高质量，使用 `diffusion_steps=20` 或更高
3. 可以使用 `inference_cfg_rate=0.7` 来提升质量

## 🔍 技术细节

### 启用的MPS优化

1. **内存管理优化**
   - 设置85%内存使用上限
   - 定期清理缓存
   - 减少内存碎片化

2. **推理参数优化**
   - 自动应用MPS友好的默认参数
   - 动态调整扩散步数
   - 优化CFG率

3. **torch.compile兼容性**
   - MPS设备使用默认编译模式
   - 提供优雅的降级处理

### 性能监控

在推理过程中，会自动打印时间统计：

```
>> gpt_gen_time: 12.34 seconds
>> gpt_forward_time: 5.67 seconds  
>> s2mel_time: 8.90 seconds
>> bigvgan_time: 2.34 seconds
>> Total inference time: 29.25 seconds
>> Generated audio length: 10.50 seconds
>> RTF: 2.786
```

## 📝 完整示例

```python
#!/usr/bin/env python3
"""
IndexTTS2 Apple Silicon 优化示例
"""

from indextts.infer_v2 import IndexTTS2

def main():
    print("=" * 60)
    print("IndexTTS2 MPS 优化示例")
    print("=" * 60)
    
    # 初始化模型
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_torch_compile=True,
    )
    
    print(f"检测到设备: {tts.device}")
    
    if tts.device == "mps":
        print("\n🚀 使用MPS优化配置")
        print("推荐参数:")
        print("  - diffusion_steps: 15 (原: 25)")
        print("  - inference_cfg_rate: 0.5 (原: 0.7)")
        print("  - max_text_tokens_per_segment: 100 (原: 120)")
    else:
        print("\n⚠️ 未检测到MPS设备，将使用标准配置")
    
    # 示例文本
    test_texts = [
        "欢迎使用IndexTTS2语音合成系统。",
        "这是针对Apple Silicon优化的语音生成示例。",
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n{'='*60}")
        print(f"生成第 {i+1} 段语音")
        print(f"文本: {text}")
        print(f"{'='*60}")
        
        # 使用MPS优化参数
        tts.infer(
            spk_audio_prompt='examples/voice_01.wav',
            text=text,
            output_path=f"gen_{i+1}.wav",
            verbose=True,
        )
    
    print("\n" + "=" * 60)
    print("所有语音生成完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

## 📚 相关资源

- [PyTorch MPS文档](https://pytorch.org/docs/stable/backends.html#mps)
- [IndexTTS2 GitHub](https://github.com/index-tts/index-tts)
- [Apple Silicon性能优化](https://developer.apple.com/metal/pytorch/)
