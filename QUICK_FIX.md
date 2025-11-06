# 快速修复指南

## 问题总结

你遇到的问题：
1. ❌ `ModuleNotFoundError: No module named 'pydantic_settings'`
2. ❌ `ImportError: email-validator is not installed`
3. ❌ `AttributeError: np.float_ was removed` (NumPy 2.0 兼容性问题)

## ✅ 已修复

所有问题已解决！已更新的文件：
- ✅ `requirements.txt` - 添加了缺失的依赖
- ✅ `fix_and_run.sh` - 自动修复和运行脚本
- ✅ `TROUBLESHOOTING.md` - 详细故障排除指南

## 🚀 现在如何运行

### 方法 1: 使用修复脚本（推荐）

```bash
cd ai-research-agent
./fix_and_run.sh
```

### 方法 2: 手动运行

```bash
cd ai-research-agent
source venv/bin/activate

# 确保所有依赖已安装
pip install -r requirements.txt

# 运行服务器
python -m uvicorn src.api.main:app --reload
```

**重要**: 使用 `python -m uvicorn` 而不是直接 `uvicorn`，确保使用虚拟环境中的正确版本。

## 📝 已修复的依赖

已添加到 `requirements.txt`：
- ✅ `email-validator>=2.0.0` - Pydantic EmailStr 验证
- ✅ `numpy<2.0` - ChromaDB 兼容性

## ✅ 验证

运行以下命令验证一切正常：

```bash
source venv/bin/activate
python -c "from src.api.main import app; print('✅ All good!')"
```

如果看到 `✅ All good!`，说明问题已解决！

## 🔍 如果仍有问题

查看 [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) 获取详细解决方案。

