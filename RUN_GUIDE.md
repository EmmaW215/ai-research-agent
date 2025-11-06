# AI Research Agent - 运行指南

本指南将帮助你快速运行 AI Research Agent 项目。

---

## 🚀 快速开始

### 首次安装（推荐使用 Python 3.12）

**方法 1: 使用自动安装脚本（推荐）**

```bash
cd ai-research-agent
./install_with_python312.sh
```

**方法 2: 手动安装**

```bash
# 1. 进入项目目录
cd ai-research-agent

# 2. 使用 Python 3.12 创建虚拟环境
python3.12 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 初始化数据库
python scripts/setup_db.py
```

### 日常使用（虚拟环境已存在）

```bash
# 1. 进入项目目录
cd ai-research-agent

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 启动服务器
uvicorn src.api.main:app --reload
```

---

## 📝 详细安装步骤

### 步骤 1: 安装 Python 3.12（如果还没有）

```bash
# macOS
brew install python@3.12

# 验证
python3.12 --version
```

### 步骤 2: 创建虚拟环境

```bash
cd ai-research-agent
python3.12 -m venv venv
source venv/bin/activate
```

### 步骤 3: 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 步骤 4: 初始化数据库（可选）

项目默认使用 SQLite，会自动创建数据库文件。如果需要初始化数据库和创建管理员用户：

```bash
python scripts/setup_db.py
```

这将创建：
- 数据库表结构
- 默认管理员用户：
  - Email: `admin@example.com`
  - Password: `admin123`
  - ⚠️ **首次登录后请立即修改密码！**

### 步骤 5: 配置环境变量（可选）

项目有默认配置，但你可以创建 `.env` 文件来自定义设置：

```bash
# 创建 .env 文件（如果需要自定义配置）
cat > .env << EOF
# Application
APP_NAME=AI Research Agent
APP_VERSION=0.1.0
ENVIRONMENT=development
DEBUG=true

# Server
HOST=0.0.0.0
PORT=8000

# Database (SQLite by default)
DATABASE_URL=sqlite:///./research_agent.db

# JWT Authentication
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=research_documents

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
EOF
```

### 步骤 6: 运行应用

**开发模式（推荐，带自动重载）：**

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

或者直接运行：

```bash
python -m src.api.main
```

**生产模式：**

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📍 访问应用

启动成功后，你可以访问：

- **API 文档（Swagger UI）**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/api/v1/health

---

## 🔧 测试 API

### 1. 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

### 2. 注册新用户

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'
```

### 3. 登录获取 Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'
```

响应示例：
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 4. 获取当前用户信息（需要 Token）

```bash
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

---

## 🧪 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并显示覆盖率
pytest --cov=src --cov-report=html

# 运行特定测试文件
pytest tests/test_auth.py -v

# 运行测试并显示详细输出
pytest -v
```

---

## 📂 项目结构

```
ai-research-agent/
├── src/
│   ├── api/              # FastAPI 路由和端点
│   │   ├── main.py      # 应用入口
│   │   └── routes/       # API 路由
│   ├── core/            # 核心业务逻辑
│   ├── adapters/        # 外部服务适配器
│   └── infrastructure/  # 基础设施（数据库、配置等）
├── tests/               # 测试文件
├── scripts/             # 工具脚本
├── requirements.txt     # Python 依赖
└── README.md            # 项目说明
```

---

## 🔍 故障排除

### 问题 1: 端口已被占用

**错误信息：**
```
ERROR:    [Errno 48] Address already in use
```

**解决方法：**
```bash
# 使用不同的端口
uvicorn src.api.main:app --reload --port 8001

# 或者查找并终止占用端口的进程
lsof -ti:8000 | xargs kill -9
```

### 问题 2: 模块导入错误

**错误信息：**
```
ModuleNotFoundError: No module named 'src'
```

**解决方法：**
```bash
# 确保在项目根目录运行
cd ai-research-agent

# 确保虚拟环境已激活
source venv/bin/activate

# 确保依赖已安装
pip install -r requirements.txt
```

### 问题 3: 数据库连接错误

**错误信息：**
```
OperationalError: unable to open database file
```

**解决方法：**
```bash
# 确保有写入权限
chmod 755 .

# 或者指定绝对路径
export DATABASE_URL=sqlite:////absolute/path/to/research_agent.db
```

### 问题 4: ChromaDB 初始化错误

**错误信息：**
```
Error creating Chroma collection
```

**解决方法：**
```bash
# 删除旧的 ChromaDB 数据（如果存在）
rm -rf chroma_db

# 重新运行应用，会自动创建新的数据库
```

---

## 📝 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DATABASE_URL` | `sqlite:///./research_agent.db` | 数据库连接 URL |
| `SECRET_KEY` | `your-secret-key-change-in-production` | JWT 密钥（生产环境必须更改） |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token 过期时间（分钟） |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | ChromaDB 存储目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `PORT` | `8000` | 服务器端口 |

---

## 🎯 快速命令参考

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python scripts/setup_db.py

# 运行开发服务器
uvicorn src.api.main:app --reload

# 运行测试
pytest

# 查看 API 文档
open http://localhost:8000/docs
```

---

## 📚 相关文档

- [项目 README](./README.md)
- [为什么需要认证？](./WHY_AUTH.md) - **了解认证的重要性**
- [Swagger UI 认证使用指南](./SWAGGER_AUTH_GUIDE.md) - **如何正确使用 Authorize 功能**
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [ChromaDB 文档](https://docs.trychroma.com/)

---

## ✅ 检查清单

在运行项目前，确认：

- [ ] Python 3.11+ 已安装
- [ ] 虚拟环境已激活
- [ ] 依赖已安装（`pip install -r requirements.txt`）
- [ ] 数据库已初始化（可选，运行 `python scripts/setup_db.py`）
- [ ] 端口 8000 未被占用
- [ ] 应用成功启动（访问 http://localhost:8000/docs 确认）

---

完成以上步骤后，你的 AI Research Agent 就可以运行了！🎉

