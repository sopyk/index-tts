## Git Push执行计划

### 修改Remote
- 将 origin 从原版仓库 (`https://github.com/index-tts/index-tts.git`) 改为你的fork仓库 (`https://github.com/sopyk/index-tts.git`)

### 添加要提交的文件
只提交变更的文件（不包括. traei目录）：
- README.md（中文版 + M4优化）
- docs/README_en.md（英文版 + M4优化）
- indextts/gpt/model_v2.py
- indextts/infer_v2.py
- indextts/s2mel/modules/commons.py

### 执行操作
1. 修改remote URL
2. 添加文件到暂存区
3. 创建commit（添加提交说明）
4. Push到远程仓库

### 预计结果
- GitHub上的sopyk/index-tts仓库会更新
- 用户可以通过 https://github.com/sopyk/index-tts 访问