# Swagger UI 认证使用指南

## 🔍 问题说明

Swagger UI 的 "Authorize" 功能出现错误的原因：

1. **OAuth2 格式不匹配**：Swagger UI 的 OAuth2 密码流期望使用 `username` 和 `password` 字段，但我们的 API 使用 `email` 和 `password` 字段。

2. **需要先登录获取 Token**：在使用 "Authorize" 功能之前，必须先通过 `/api/v1/auth/login` 端点获取 access token。

---

## ✅ 正确的使用方法

### 方法 1: 使用 Swagger UI 的 Authorize 功能（推荐）

#### 步骤 1: 确保已注册用户

如果还没有用户，先注册：

1. 在 Swagger UI 中找到 **`POST /api/v1/auth/register`** 端点
2. 点击 "Try it out"
3. 填写注册信息：
   ```json
   {
     "email": "user@example.com",
     "password": "SecurePass123!",
     "full_name": "John Doe"
   }
   ```
4. 点击 "Execute" 完成注册

#### 步骤 2: 登录获取 Token

1. 在 Swagger UI 中找到 **`POST /api/v1/auth/login`** 端点
2. 点击 "Try it out"
3. 填写登录信息：
   ```json
   {
     "email": "user@example.com",
     "password": "SecurePass123!"
   }
   ```
4. 点击 "Execute"
5. 在响应中找到 `access_token`，例如：
   ```json
   {
     "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
     "token_type": "bearer",
     "expires_in": 1800
   }
   ```
6. **复制整个 `access_token` 的值**（不包括引号）

#### 步骤 3: 使用 Authorize 功能

1. 点击页面右上角的绿色 **"Authorize"** 按钮
2. 在弹出的对话框中，你会看到 **"Bearer (http, bearer)"** 安全方案
3. 在 **"Value"** 输入框中，**直接粘贴刚才复制的 `access_token`**（不需要添加 "Bearer " 前缀）
4. 点击 **"Authorize"** 按钮
5. 点击 **"Close"** 关闭对话框

**✅ 现在你已成功认证！** 所有需要认证的端点都会自动包含这个 token。

#### 步骤 4: 测试认证端点

1. 找到 **`GET /api/v1/auth/me`** 端点
2. 点击 "Try it out"
3. 点击 "Execute"
4. 应该能成功返回当前用户信息

---

### 方法 2: 使用 curl 命令（命令行）

#### 步骤 1: 注册用户

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "full_name": "John Doe"
  }'
```

#### 步骤 2: 登录获取 Token

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
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### 步骤 3: 使用 Token 访问受保护的端点

```bash
# 将 YOUR_TOKEN_HERE 替换为实际的 access_token
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

---

## 🐛 常见错误及解决方法

### 错误 1: "Auth error: Error: Unprocessable Entity"

**原因：**
- 在 Authorize 对话框中输入了错误的格式
- Token 格式不正确或已损坏

**解决方法：**
- 确保只粘贴 token 值（不包括 "Bearer " 前缀）
- 确保复制完整的 token（从响应中完整复制，不要截断）
- 先通过 `/api/v1/auth/login` 端点获取新的 token，然后粘贴到 Authorize 对话框

### 错误 2: "Could not validate credentials"

**原因：**
- Token 已过期（默认 30 分钟）
- Token 无效或格式错误

**解决方法：**
- 重新登录获取新的 token
- 确保复制完整的 token（包括所有字符）

### 错误 3: "User account is inactive"

**原因：**
- 用户账户被禁用

**解决方法：**
- 检查数据库中的用户状态
- 确保用户 `is_active` 字段为 `true`

---

## 📝 快速参考

### 默认管理员账户

如果使用 `scripts/setup_db.py` 初始化数据库，会创建默认管理员：

- **Email**: `admin@example.com`
- **Password**: `admin123`
- ⚠️ **首次登录后请立即修改密码！**

### Token 有效期

- 默认：30 分钟（1800 秒）
- 可在 `.env` 文件中配置：`ACCESS_TOKEN_EXPIRE_MINUTES=30`

### 测试完整流程

```bash
# 1. 注册
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test123!","full_name":"Test User"}'

# 2. 登录
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test123!"}' \
  | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

# 3. 获取用户信息
curl -X GET http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

---

## ✅ 检查清单

使用 Swagger UI 认证前，确认：

- [ ] 服务器正在运行（http://localhost:8000）
- [ ] 已注册用户或使用默认管理员账户
- [ ] 已通过 `/api/v1/auth/login` 获取 access token
- [ ] 在 Authorize 对话框中粘贴了完整的 token（只粘贴 token 值，不包括 "Bearer " 前缀）
- [ ] Token 未过期（30 分钟内）
- [ ] Authorize 对话框中显示的是 "Bearer (http, bearer)" 而不是 OAuth2 密码流

---

完成以上步骤后，你就可以在 Swagger UI 中正常使用需要认证的端点了！🎉

