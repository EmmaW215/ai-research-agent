# 端口 8000 被占用错误修复指南

## 🔍 错误原因

**错误信息**: `ERROR: [Errno 48] Address already in use`

**原因**: 端口 8000 已经被另一个进程占用。这通常发生在：
- 之前启动的 uvicorn 服务器还在运行
- 其他应用程序正在使用端口 8000
- 之前的进程没有正确关闭

## ✅ 快速解决方案

### 方法 1: 使用更新后的修复脚本（推荐）

我已经更新了 `fix_and_run.sh`，现在会自动检测并清理占用端口的进程：

```bash
cd ai-research-agent
./fix_and_run.sh
```

### 方法 2: 手动停止占用端口的进程

```bash
# 查找占用端口 8000 的进程
lsof -ti:8000

# 停止所有占用端口 8000 的进程
lsof -ti:8000 | xargs kill -9

# 验证端口已释放
lsof -ti:8000 || echo "Port 8000 is now free"
```

### 方法 3: 使用不同的端口

如果不想停止现有进程，可以使用其他端口：

```bash
source venv/bin/activate
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001
```

然后访问: http://localhost:8001/docs

## 🔧 详细步骤

### 步骤 1: 检查端口占用

```bash
# 查看占用端口 8000 的进程
lsof -i:8000

# 或者只获取进程 ID
lsof -ti:8000
```

### 步骤 2: 停止进程

```bash
# 方法 A: 使用进程 ID（替换 PID 为实际进程号）
kill -9 <PID>

# 方法 B: 一行命令停止所有占用 8000 端口的进程
lsof -ti:8000 | xargs kill -9
```

### 步骤 3: 验证端口已释放

```bash
lsof -ti:8000 || echo "✅ Port 8000 is free"
```

### 步骤 4: 重新启动服务器

```bash
source venv/bin/activate
python -m uvicorn src.api.main:app --reload
```

## 📝 预防措施

### 1. 使用更新后的脚本

`fix_and_run.sh` 现在会自动：
- ✅ 检测端口占用
- ✅ 自动停止占用端口的进程
- ✅ 验证端口已释放
- ✅ 然后启动服务器

### 2. 正确停止服务器

当需要停止服务器时：
- 在运行服务器的终端按 `Ctrl+C`
- 确保进程完全退出

### 3. 检查后台进程

如果怀疑有后台进程在运行：

```bash
# 查看所有 uvicorn 进程
ps aux | grep uvicorn | grep -v grep

# 查看所有 Python 进程
ps aux | grep python | grep uvicorn
```

## 🛠️ 常用命令

```bash
# 检查端口占用
lsof -i:8000

# 停止占用端口的进程
lsof -ti:8000 | xargs kill -9

# 查看所有 uvicorn 进程
ps aux | grep uvicorn

# 停止所有 uvicorn 进程（谨慎使用）
pkill -f uvicorn
```

## ⚠️ 注意事项

1. **不要强制杀死重要进程**: 确保你杀死的是你自己的开发服务器进程
2. **检查进程**: 在杀死进程前，可以使用 `ps aux | grep <PID>` 查看进程详情
3. **使用不同端口**: 如果端口被重要应用占用，考虑使用其他端口（如 8001, 8002）

## ✅ 验证修复

修复后，运行：

```bash
./fix_and_run.sh
```

应该看到：
- ✅ Port 8000 is available（或已自动清理）
- ✅ 服务器成功启动
- ✅ 可以访问 http://localhost:8000/docs

