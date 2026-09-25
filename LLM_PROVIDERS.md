# LLM 提供商配置

本项目支持多个 LLM 提供商，通过环境变量 `LLM_PROVIDER` 灵活切换。

## 支持的提供商

### 1. Grok / xAI（OpenAI 兼容接口）

```bash
LLM_PROVIDER=grok
GROK_API_KEY=你的Grok或xAI_API_Key
GROK_BASE_URL=https://api.x.ai/v1
GROK_MODEL=grok-4.6
GROK_USE_RESPONSES_API=1
GROK_REASONING_EFFORT=low
```

如果使用 Grok CLI 的第三方兼容端点，`GROK_BASE_URL` 改成 CLI 配置中的地址，例如 `https://www.micuapi.ai/v1`。项目不会自动读取 CLI 配置文件。

官方 xAI API 文档：https://docs.x.ai/

---

### 2. OpenAI

使用 OpenAI Responses API 和原生结构化输出。

```bash
# 在 .env.local 中配置
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxx
OPENAI_MODEL=gpt-5.6-terra
# TRAVELMIND_OPENAI_REASONING_EFFORT=none  # 可选，默认 none
```

API Key 获取：https://platform.openai.com/api-keys

`gpt-5.6-terra` 不支持 Free API tier，请确认 API 账户已启用付费额度。ChatGPT/Codex 订阅不等同于 API 额度。

---

### 3. DeepSeek（默认）

高性能、低成本的国产大模型。

**配置步骤：**

```bash
# 在 .env.local 中配置
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk_xxxxxxxxxxxx
# DEEPSEEK_MODEL=deepseek-v4-flash  # 可选，默认值已设置
```

**可用模型：**
- `deepseek-v4-flash`（默认，快速）
- `deepseek-v4-pro`（更强大）

**API 获取：**
https://platform.deepseek.com

---

### 4. 豆包（Doubao）

字节跳动出品的大模型，通过 Volces Ark 平台提供。

**配置步骤：**

```bash
# 在 .env.local 中配置
LLM_PROVIDER=doubao
DOUBAO_API_KEY=xxxxxxxxxxxxxxxx
# DOUBAO_MODEL=doubao-pro-4k  # 可选，默认值已设置
```

**可用模型：**
- `doubao-pro-4k`（默认）
- `doubao-vision-pro-4k`（支持视觉）

**API 获取：**
https://console.volcengine.com/ark

**注意：** 豆包 API 需要在 Volces Ark 控制台创建"推理端点"后，使用对应的 endpoint ID。配置示例：
```bash
DOUBAO_API_KEY=your_endpoint_id
```

---

## 工作原理

所有 LLM 调用通过统一的工厂函数完成：

- `app/llm/factory.py` — 工厂函数（自动选择提供商）
- `app/llm/openai.py` — OpenAI Responses API 客户端实现
- `app/llm/grok.py` — Grok/xAI OpenAI 兼容客户端实现
- `app/llm/deepseek.py` — DeepSeek 客户端实现
- `app/llm/doubao.py` — 豆包客户端实现

核心函数：
- `build_chat_llm()` — 创建聊天客户端
- `build_structured_llm()` — 创建结构化输出客户端

在 `nodes.py`、`profile_updater.py` 等调用点，统一使用工厂函数，提供商自动切换。

### 可选的任务级模型路由

默认关闭，不影响现有调用：

```bash
LLM_ROUTING_ENABLED=1
LLM_SMALL_MODEL=当前供应商可访问的低成本模型名
LLM_ROUTING_SHORT_TOKEN_THRESHOLD=1200
```

Intent、Query Rewrite、画像/语义记忆抽取等有界任务使用小模型；Planner、Reviewer 和 Judge 保留主模型。路由不会跨供应商，`LLM_SMALL_MODEL` 必须属于当前 `LLM_PROVIDER`。项目目前只验证了离线路由契约，真实成本、延迟与质量仍需在同一评测集和同一供应商条件下做 A/B 后再下结论。

---

## 切换提供商

**方法 1：环境变量**
```bash
export LLM_PROVIDER=openai
python run.py
```

**方法 2：.env.local 配置**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.6-terra
```

**方法 3：容器部署**
```bash
docker run -e LLM_PROVIDER=openai -e OPENAI_API_KEY=xxx -e OPENAI_MODEL=gpt-5.6-terra my_app
```

---

## 性能对比

| 维度 | OpenAI | Grok | DeepSeek | 豆包 |
|------|--------|------|----------|------|
| 成本 | 中高 | 取决于端点 | 低 | 中 |
| 速度 | 快 | 取决于端点 | 快 | 中等 |
| 质量 | 优秀 | 优秀 | 优秀 | 优秀 |
| 国内网络 | 可能需要代理 | 官方/中转差异大 | 视网络而定 | 通常较低延迟 |

---

## 故障排查

### 1. 提供商不支持

```
ValueError: 未知的 LLM 提供商：xxx，支持：openai, grok, deepseek, doubao
```

**解决：** 检查 `LLM_PROVIDER` 环境变量，确保值为 `openai`、`grok`、`deepseek` 或 `doubao`。

### 2. API Key 缺失

```
RuntimeError: 缺少 DOUBAO_API_KEY。请在 .env.local 中配置后重试。
```

**解决：** 
- 确认 API Key 已在 `.env.local` 中配置
- 重启应用以加载新的环境变量
- 检查 Key 格式是否正确（无多余空格）

### 3. 连接失败

如果遇到网络问题，可检查：
- VPN/代理配置（参见 `HTTPS_PROXY` 环境变量）
- API 端点是否可访问
- 防火墙配置

---

## 开发指南

### 添加新的 LLM 提供商

1. **创建提供商模块** `app/llm/newprovider.py`：
   ```python
   def build_chat_newprovider(*, model: str | None = None, temperature: float = 0) -> Any:
       # 实现 ChatOpenAI 兼容的客户端
       ...

   def build_structured_newprovider(schema: type[SchemaT], ...) -> Any:
       # 实现结构化输出
       ...
   ```

2. **更新工厂函数** `app/llm/factory.py`：
   ```python
   def resolve_llm_provider() -> LLMProvider:
       # 在类型注解中添加新提供商
       # 在验证逻辑中添加支持
       ...

   def build_structured_llm(...):
       if provider == "newprovider":
           from app.llm.newprovider import build_structured_newprovider
           return build_structured_newprovider(...)
   ```

3. **更新文档** 本文件，记录新提供商的配置方式

---

## 使用建议

- **开发环境**：根据账户条件使用 `gpt-5.6-terra` 或成本较低的提供商
- **生产环境**：结合质量、延迟、成本与国内网络条件选择提供商
- **A/B 测试**：使用工厂函数易于在两个提供商间快速切换
