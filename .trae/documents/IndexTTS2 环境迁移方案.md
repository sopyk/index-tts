# IndexTTS2 迁移到 ~/app/ 完整操作计划

## 目标
- 退出conda base环境
- 将项目移动到 ~/app/index-tts/
- 使用uv安装依赖到系统级

## 步骤

### 步骤1：退出conda环境
```bash
conda deactivate
```
退出后，终端前缀应该变成空白（系统Python）

### 步骤2：移动项目到新位置
```bash
# 创建目标目录
mkdir -p ~/app

# 移动项目（保留权限和链接）
mv /Users/song/index-tts ~/app/

# 验证移动成功
ls -la ~/app/
```

### 步骤3：安装uv并同步依赖
```bash
# 安装uv（如未安装）
pip install -U uv

# 进入项目目录
cd ~/app/index-tts

# 安装所有依赖
uv sync --all-extras
```

### 步骤4：验证安装成功
```bash
cd ~/app/index-tts
uv run python -c "from indextts.infer_v2 import IndexTTS2; print('✓ 安装成功')"
```

### 步骤5：可选 - 清理conda（谨慎）
```bash
# 仅移除conda的激活脚本引用（不删除conda本身）
# 编辑 ~/.zshrc 或 ~/.bash_profile
# 注释或删除: conda initialize...
```

## 验证检查点
- [ ] 终端不再显示 `(base)` 或 `(env_name)` 前缀
- [ ] 项目在 `~/app/index-tts/`
- [ ] `uv run python` 能正常导入indextts
- [ ] MPS加速正常工作（根据README优化）

## 注意事项
- 移动项目后，checkpoints目录如果使用软链接需要重新检查
- 确保新路径有足够的磁盘空间
- 保持项目权限正确