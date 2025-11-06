# 虚拟环境错误修复指南

## 🔍 问题原因

**错误**: `ModuleNotFoundError: No module named 'fastapi'`

**原因**: 你使用了**错误的虚拟环境**！

从错误日志可以看到：
- ❌ 使用的是父项目的虚拟环境（Python 3.14）：`VicProject1_AIResearchAgent/venv`
- ✅ 应该使用 `ai-research-agent` 项目的虚拟环境（Python 3.12）：`ai-research-agent/venv`

## ✅ 解决方案

### 方法 1: 确保在正确的目录和虚拟环境

```bash
# 1. 进入 ai-research-agent 目录
cd ai-research-agent

# 2. 如果之前激活了父项目的虚拟环境，先退出
deactivate  # 如果有的话

# 3. 激活 ai-research-agent 的虚拟环境
source venv/bin/activate

# 4. 验证 Python 版本和路径
python --version  # 应该显示 Python 3.12.x
which python      # 应该指向 ai-research-agent/venv/bin/python

# 5. 验证 fastapi 已安装
python -c "import fastapi; print('✅ FastAPI installed')"

# 6. 运行服务器
python -m uvicorn src.api.main:app --reload
```

### 方法 2: 使用修复脚本（推荐）

```bash
cd ai-research-agent
./fix_and_run.sh
```

脚本会自动：
- ✅ 退出任何已激活的虚拟环境
- ✅ 激活正确的虚拟环境（ai-research-agent/venv）
- ✅ 验证 Python 版本和路径
- ✅ 安装依赖
- ✅ 启动服务器

## 🔍 如何识别正确的虚拟环境

### 正确的虚拟环境（应该使用）：
```
Python: .../ai-research-agent/venv/bin/python
Python version: Python 3.12.x
Virtual env: .../ai-research-agent/venv
```

### 错误的虚拟环境（不要使用）：
```
Python: .../VicProject1_AIResearchAgent/venv/bin/python
Python version: Python 3.14.x
Virtual env: .../VicProject1_AIResearchAgent/venv
```

## 📋 检查清单

运行前检查：

```bash
# 1. 确认当前目录
pwd
# 应该显示: .../ai-research-agent

# 2. 检查虚拟环境
echo $VIRTUAL_ENV
# 应该显示: .../ai-research-agent/venv

# 3. 检查 Python 路径
which python
# 应该显示: .../ai-research-agent/venv/bin/python

# 4. 检查 Python 版本
python --version
# 应该显示: Python 3.12.x

# 5. 检查 fastapi
python -c "import fastapi; print('✅ FastAPI OK')"
```

## 🛠️ 如果仍然有问题

### 完全重新设置：

```bash
# 1. 退出所有虚拟环境
deactivate 2>/dev/null || true

# 2. 进入项目目录
cd ai-research-agent

# 3. 删除旧的虚拟环境（如果需要）
rm -rf venv

# 4. 重新创建虚拟环境
python3.12 -m venv venv

# 5. 激活虚拟环境
source venv/bin/activate

# 6. 安装依赖
pip install -r requirements.txt

# 7. 运行
python -m uvicorn src.api.main:app --reload
```

## ⚠️ 重要提示

1. **每个项目有独立的虚拟环境**：
   - 父项目：`VicProject1_AIResearchAgent/venv` (Python 3.14) - 用于 Claude Agent SDK
   - 子项目：`ai-research-agent/venv` (Python 3.12) - 用于 AI Research Agent

2. **不要混用虚拟环境**：
   - 运行 `ai-research-agent` 时，必须使用 `ai-research-agent/venv`
   - 不要在父项目目录运行 `ai-research-agent` 的命令

3. **使用完整路径启动**（如果仍有问题）：
```bash
cd ai-research-agent
./venv/bin/python -m uvicorn src.api.main:app --reload
```

