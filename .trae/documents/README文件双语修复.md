## README文件修复计划

### 目标
- **根目录README.md**：中文版 + 中文M4优化说明
- **docs/README_en.md**：英文版 + 英文M4优化说明

### 具体步骤

#### 步骤1：为根目录README.md添加中文M4优化部分
- 在第1行后添加分隔线 "---"
- 添加docs/README_en.md中第4-154行的中文M4优化内容
- 添加另一个分隔线 "---"

#### 步骤2：将docs/README_en.md的M4优化部分翻译成英文
- 将第4-154行的中文M4优化翻译成英文（保留代码块不变）
- 确保标题、语言都改为英文

#### 步骤3：修正链接引用
**docs/README_en.md中的链接**：
- 修改语言切换链接：
  - 简体中文 → README_zh.md（保持不变）
  - English → README.md（保持不变）
- 修改图片路径：assets/ → ../assets/

**根目录README.md中的链接**：
- 简体中文 → docs/README_en.md（因为docs下已有README_zh.md）
- English → docs/README_en.md（指向英文版）
- 修改图片路径：../assets/ → assets/