## 修复方案

**问题：** 调用 `generate()` 时传递了不被支持的参数：`diffusion_steps`, `inference_cfg_rate`, `max_text_tokens_per_segment`

**修改文件：**
- `indextts/gpt/model_v2.py`

**修改内容：**
在 `inference_speech()` 方法中（第778行附近），在调用 `self.inference_model.generate()` 之前，过滤掉 `hf_generate_kwargs` 中不被支持的参数：

```python
# 在第777行后添加参数过滤
hf_generate_kwargs_filtered = {
    k: v for k, v in hf_generate_kwargs.items() 
    if k not in ['diffusion_steps', 'inference_cfg_rate', 'max_text_tokens_per_segment']
}

# 将第778行的 **hf_generate_kwargs 改为 **hf_generate_kwargs_filtered
```

**优点：**
- 简单直接，不影响其他代码逻辑
- 明确过滤掉特定的不兼容参数
- 保持向后兼容性