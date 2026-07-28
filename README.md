# TravelMind-Agent：多智能体旅行规划助手

> 基于开源项目 [FloatTrip](https://github.com/shouzhuoshouzhuo/FloatTrip) 的课程二次开发。面向“输入旅行需求，生成可解释的行程方案”的场景，重点实现多智能体协作、工具调用、个性化记忆与可复现测试。

## 项目能力

- 解析目的地、日期、预算、偏好等旅行约束，缺少关键字段时提示用户补充。
- 调用高德 POI、天气、路径等服务，生成景点候选、餐饮、酒店与出行建议。
- 使用 Planner / Reviewer 工作流生成并审查行程；候选池中没有用户指定景点时可重新检索。
- 支持用户画像与语义记忆，降低重复表达偏好的成本。
- 前端展示规划过程、景点候选、行程路线、预算与阶段性播报。
- 提供独立的课程实验三多智能体核心：Supervisor 集中调度、Agent 私有记忆、工具权限隔离与失败重试。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 后端 | Python、FastAPI、Uvicorn、Pydantic |
| Agent 编排 | LangGraph；实验三核心使用自定义 Supervisor |
| 大模型 | DeepSeek / 豆包（环境变量切换） |
| 地图与工具 | 高德地图 Web 服务 API、JS API |
| 数据与记忆 | SQLite、Chroma 语义记忆（本地运行数据不提交） |
| 前端 | React/JSX、SSE、Amap JS API |
| 质量保障 | pytest、离线 Golden Set、LLM-as-Judge 评测 |

## 架构概览

```text
用户需求
  -> Intent / Query Rewrite
  -> POI 与天气工具
  -> Planner Agent
  -> Reviewer Agent（不通过则有限次返工）
  -> Time Check / 餐饮推荐 / 路线与预算汇总
  -> 流式展示行程

实验三独立核心：
Supervisor -> IntentAgent -> POIResearchAgent -> PlannerAgent -> ReviewerAgent
```

实验三核心中，每个 Worker Agent 都拥有独立的 `system_prompt` 与 `private_memory`。Agent 仅通过结构化 `AgentMessage` 传递任务结果，不能读取其他 Agent 的私有记忆；POI 工具也通过白名单限制为 POIResearchAgent 使用。

## 快速开始（Windows）

### 1. 克隆并创建虚拟环境

```powershell
git clone https://github.com/yf-922/TravelMind-Agent.git
cd TravelMind-Agent
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env.local`，按需填写：

```env
AMAP_API_KEY=你的高德Web服务Key
AMAP_JS_KEY=你的高德JS_API_Key
AMAP_JS_SECURITY_CODE=你的高德安全密钥
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的DeepSeek_Key
```

`.env.local`、本地数据库和记忆数据均被 `.gitignore` 排除，切勿提交真实 Key。

### 3. 启动 Web 应用

```powershell
.\.venv\Scripts\python.exe run.py
```

打开 <http://127.0.0.1:8765>。

## 实验三：可运行的多智能体核心

课程实验三的独立实现位于 [`app/multi_agent_core`](app/multi_agent_core)。使用离线 Fixture 数据即可稳定复现，不依赖 API Key：

```powershell
.\.venv\Scripts\python.exe -m app.multi_agent_core.demo_run --offline --city Beijing --request "Plan a relaxed cultural day trip"
```

终端会输出：

1. `dispatch_log`：Supervisor 依次调度四个 Agent 的证据；
2. `PRIVATE MEMORY CONTENTS`：各 Agent 系统提示与私有记忆不同的证据；
3. POI 工具调用和最终行程。

运行实验三测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_multi_agent_core.py -q
```

测试覆盖：多 Worker 调度、POI 候选约束、Reviewer 驳回后的返工路由、超时重试与结构化失败、跨 Agent 工具权限拒绝、模型适配器私有上下文与 SQLite 记忆隔离。

## 项目结构

```text
app/
  api/                 # FastAPI 路由
  planning/            # LangGraph 旅行规划工作流
  providers/           # 高德 POI、天气、路径等外部服务
  multi_agent_core/    # 实验三：独立可运行的 Supervisor 与 Worker Agent
  core/                # 配置、数据库、缓存、记忆等基础能力
frontend/              # React/JSX 前端
tests/                 # 单元、集成、端到端和评测测试
```

## 测试与评估

```powershell
# 多智能体课程核心测试
.\.venv\Scripts\python.exe -m pytest tests\test_multi_agent_core.py -q

# 项目完整测试（按本地依赖与 Key 配置执行）
.\.venv\Scripts\python.exe -m pytest -q
```

离线测试用于验证结构化输出、调度、隔离与降级路径；涉及真实地图、天气、票价的数据必须在配置 Key 后实时调用，不能将本地测试数据当作实时事实。

## 说明与致谢

- 本项目用于《人工智能应用实践》课程实验和个人工程学习。
- 原始项目：[`shouzhuoshouzhuo/FloatTrip`](https://github.com/shouzhuoshouzhuo/FloatTrip)。二次开发内容应以本仓库提交记录为准。
- 请遵守高德地图、模型服务商及第三方服务的使用条款，不要提交 API Key、用户对话、用户画像或本地数据库。
