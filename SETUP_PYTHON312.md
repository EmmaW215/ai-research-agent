# Python 3.12 安装指南

## 🎯 为什么需要 Python 3.12？

Python 3.14 是最新版本，但许多包（如 `chromadb`、`pydantic-core`）还没有官方支持。使用 Python 3.12 可以确保所有依赖包都能正常安装和工作。

---

## 📦 安装 Python 3.12

### macOS (使用 Homebrew)

```bash
# 安装 Python 3.12
brew install python@3.12

# 验证安装
python3.12 --version
```

### 其他系统

访问 [Python 官网](https://www.python.org/downloads/) 下载 Python 3.12.x

---

## 🚀 使用安装脚本（推荐）

### 方法 1: 自动安装脚本

```bash
cd ai-research-agent
./install_with_python312.sh
```

这个脚本会自动：
- ✅ 检测 Python 3.12
- ✅ 创建虚拟环境
- ✅ 安装所有依赖
- ✅ 初始化数据库
- ✅ 验证安装

### 方法 2: 手动安装

```bash
# 1. 进入项目目录
cd ai-research-agent

# 2. 删除旧的虚拟环境（如果存在）
rm -rf venv

# 3. 使用 Python 3.12 创建虚拟环境
python3.12 -m venv venv

# 4. 激活虚拟环境
source venv/bin/activate

# 5. 升级 pip
pip install --upgrade pip

# 6. 安装依赖
pip install -r requirements.txt

# 7. 初始化数据库
python scripts/setup_db.py

# 8. 验证安装
python -c "import fastapi, sqlalchemy, chromadb; print('✅ All packages installed')"
```

---

## ✅ 验证安装

```bash
# 激活虚拟环境
source venv/bin/activate

# 检查 Python 版本
python --version
# 应该显示: Python 3.12.x

# 检查关键包
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import sqlalchemy; print('SQLAlchemy:', sqlalchemy.__version__)"
python -c "import chromadb; print('ChromaDB installed')"
```

---

## 🎯 启动应用

安装完成后：

```bash
# 激活虚拟环境
source venv/bin/activate

# 启动服务器
uvicorn src.api.main:app --reload
```

访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/v1/health

---

## 🔍 故障排除

### 问题 1: 找不到 python3.12

**解决方案：**
```bash
# macOS
brew install python@3.12

# 或者检查是否有其他版本
ls /usr/local/bin/python3.*
```

### 问题 2: 安装脚本失败

**解决方案：**
1. 确保 Python 3.12 已安装
2. 检查脚本权限：`chmod +x install_with_python312.sh`
3. 手动执行安装步骤

### 问题 3: 依赖包仍然安装失败

**解决方案：**
```bash
# 尝试单独安装有问题的包
pip install --upgrade pip setuptools wheel
pip install chromadb --no-cache-dir
```

---

## 📝 注意事项

1. **Python 版本兼容性**：
   - ✅ Python 3.11, 3.12, 3.13 - 完全支持
   - ⚠️ Python 3.14+ - 部分包可能不兼容

2. **虚拟环境**：
   - 每个项目应该有自己的虚拟环境
   - 不要将虚拟环境提交到 Git

3. **依赖管理**：
   - `requirements.txt` 包含所有必需依赖
   - `requirements-postgresql.txt` 包含可选的 PostgreSQL 支持

---

## 🎉 完成！

安装完成后，你的项目应该可以正常运行了！

如果遇到任何问题，请查看 [INSTALL_FIX.md](./INSTALL_FIX.md) 获取详细的问题解决方案。

