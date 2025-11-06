# 🚀 快速开始 - AI Research Agent

## 方法 1: 使用快速启动脚本（推荐）

```bash
cd ai-research-agent
./quick_start.sh
```

这个脚本会自动：
- ✅ 检查 Python 版本
- ✅ 创建虚拟环境（如果不存在）
- ✅ 安装所有依赖
- ✅ 初始化数据库
- ✅ 创建 .env 文件（如果需要）
- ✅ 启动开发服务器

---

## 方法 2: 手动步骤

### 1. 进入项目目录

```bash
cd ai-research-agent
```

### 2. 创建并激活虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 初始化数据库（可选）

```bash
python scripts/setup_db.py
```

这将创建默认管理员账户：
- Email: `admin@example.com`
- Password: `admin123`

### 5. 启动服务器

```bash
uvicorn src.api.main:app --reload
```

---

## 📍 访问应用

启动后访问：

- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/v1/health

---

## 🧪 测试 API

### 注册用户

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "Test123!", "full_name": "Test User"}'
```

### 登录

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "Test123!"}'
```

---

## 📚 详细文档

查看 [RUN_GUIDE.md](./RUN_GUIDE.md) 获取完整说明。

