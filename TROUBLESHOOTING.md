# 故障排除指南

## 问题：ModuleNotFoundError: No module named 'pydantic_settings'

### 原因
依赖包没有正确安装或版本不兼容。

### 解决方案

**方法 1: 使用修复脚本（推荐）**

```bash
./fix_and_run.sh
```

**方法 2: 手动修复**

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 重新安装所有依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "from src.api.main import app; print('✅ Success')"
```

---

## 问题：AttributeError: `np.float_` was removed in NumPy 2.0

### 原因
ChromaDB 与 NumPy 2.0 不兼容，需要使用 NumPy < 2.0。

### 解决方案

```bash
source venv/bin/activate
pip install "numpy<2.0" --force-reinstall
```

已更新 `requirements.txt` 包含此限制。

---

## 问题：ImportError: email-validator is not installed

### 原因
Pydantic 的 EmailStr 验证需要 `email-validator` 包。

### 解决方案

```bash
source venv/bin/activate
pip install email-validator
```

已更新 `requirements.txt` 包含此依赖。

---

## 问题：uvicorn 不断重载并报错

### 原因
1. 虚拟环境未正确激活
2. 使用了系统级别的 uvicorn 而不是虚拟环境中的
3. 依赖包缺失

### 解决方案

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 使用 python -m uvicorn 而不是直接 uvicorn
python -m uvicorn src.api.main:app --reload

# 或者使用修复脚本
./fix_and_run.sh
```

---

## 完整修复步骤

如果遇到多个问题，按以下步骤修复：

```bash
# 1. 进入项目目录
cd ai-research-agent

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 重新安装所有依赖
pip install -r requirements.txt

# 5. 验证关键包
python -c "from pydantic_settings import BaseSettings; print('✅ pydantic_settings')"
python -c "import email_validator; print('✅ email_validator')"
python -c "import numpy; print(f'✅ numpy {numpy.__version__}')"
python -c "from src.api.main import app; print('✅ App imports successfully')"

# 6. 运行应用
python -m uvicorn src.api.main:app --reload
```

---

## 常见错误检查清单

- [ ] 虚拟环境已激活（终端提示符前有 `(venv)`）
- [ ] `pydantic-settings` 已安装
- [ ] `email-validator` 已安装
- [ ] `numpy<2.0` 已安装（如果使用 chromadb）
- [ ] 使用 `python -m uvicorn` 而不是直接 `uvicorn`
- [ ] 在项目根目录运行命令

---

## 已验证的依赖版本

以下版本组合已验证可以正常工作：

- Python: 3.12
- pydantic: 2.5.0 (或 2.12.4)
- pydantic-settings: 2.1.0
- email-validator: >=2.0.0
- numpy: <2.0 (推荐 1.26.4)
- chromadb: >=0.4.22

---

## 如果问题仍然存在

1. **删除并重新创建虚拟环境**：
```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **检查 Python 版本**：
```bash
python --version  # 应该是 3.12.x
```

3. **查看详细错误信息**：
```bash
python -c "from src.api.main import app" 2>&1 | head -20
```

