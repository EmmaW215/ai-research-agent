# 安装问题修复指南

## 问题分析

你遇到了两个主要问题：

1. **`psycopg2-binary` 安装失败** - 项目默认使用 SQLite，不需要 PostgreSQL
2. **`chromadb` 与 Python 3.14 不兼容** - Python 3.14 太新，许多包还未支持

## ✅ 解决方案

### 方案 1: 使用 Python 3.11 或 3.12（推荐）

Python 3.14 是最新版本，许多包（如 chromadb、onnxruntime）还没有官方支持。

**推荐使用 Python 3.11 或 3.12**：

```bash
# 1. 安装 Python 3.12（如果还没有）
# macOS with Homebrew:
brew install python@3.12

# 2. 创建新的虚拟环境，指定 Python 版本
cd ai-research-agent
python3.12 -m venv venv
source venv/bin/activate

# 3. 安装依赖（现在应该可以成功）
pip install --upgrade pip
pip install -r requirements.txt
```

### 方案 2: 暂时移除可选依赖（快速修复）

如果暂时不需要向量存储功能，可以暂时移除 chromadb：

```bash
# 1. 编辑 requirements.txt，注释掉 chromadb
# 2. 安装其他依赖
pip install -r requirements.txt --no-deps
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 pydantic-settings==2.1.0 sqlalchemy==2.0.23 alembic==1.13.0 python-jose[cryptography]==3.3.0 passlib[bcrypt]==1.7.4 python-multipart==0.0.6 pytest==7.4.3 pytest-asyncio==0.21.1 pytest-cov==4.1.0 httpx==0.25.2 python-dotenv==1.0.0
```

### 方案 3: 使用更新版本的依赖（可能不兼容）

尝试使用更新的包版本，但可能与其他依赖冲突：

```bash
pip install --upgrade fastapi uvicorn pydantic chromadb
```

---

## 🎯 已修复的问题

我已经修复了 `requirements.txt`：

1. ✅ **移除了 `psycopg2-binary`** - 因为默认使用 SQLite，不需要 PostgreSQL
2. ✅ **更新了 `chromadb` 版本要求** - 改为 `>=0.4.22`，但 Python 3.14 仍可能不兼容

---

## 📋 推荐的安装步骤

### 步骤 1: 检查 Python 版本

```bash
python3 --version
```

如果显示 Python 3.14，建议降级到 3.12：

```bash
# macOS
brew install python@3.12
```

### 步骤 2: 创建新虚拟环境（使用 Python 3.12）

```bash
cd ai-research-agent
rm -rf venv  # 删除旧的虚拟环境
python3.12 -m venv venv
source venv/bin/activate
```

### 步骤 3: 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 步骤 4: 验证安装

```bash
python -c "import fastapi; import sqlalchemy; print('✅ Core dependencies installed')"
```

---

## 🔍 如果仍然遇到问题

### 问题 A: chromadb 仍然无法安装

**临时解决方案** - 注释掉 chromadb，先运行基础功能：

```bash
# 编辑 requirements.txt，注释掉 chromadb 行
# 然后安装
pip install -r requirements.txt
```

项目的基础功能（API、认证、数据库）不需要 chromadb。向量存储功能可以稍后添加。

### 问题 B: 其他依赖冲突

```bash
# 尝试不固定版本安装
pip install fastapi uvicorn[standard] pydantic pydantic-settings sqlalchemy alembic python-jose[cryptography] passlib[bcrypt] python-multipart pytest pytest-asyncio pytest-cov httpx python-dotenv
```

---

## 📝 当前 requirements.txt 状态

已更新的 `requirements.txt`：
- ✅ 移除了 `psycopg2-binary`（可选，仅 PostgreSQL 需要）
- ✅ 更新了 `chromadb` 版本要求（但 Python 3.14 仍可能不兼容）

**建议**: 使用 Python 3.12 重新创建虚拟环境。

