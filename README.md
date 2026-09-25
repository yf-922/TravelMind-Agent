# TravelMind-Agent：LLM 多 Agent 旅行规划助手

> 基于开源项目 [FloatTrip](https://github.com/shouzhuoshouzhuo/FloatTrip) 的课程二次开发。面向“输入旅行需求，生成可解释的行程方案”的场景，重点实现多智能体协作、工具调用、个性化记忆与可复现测试。

## 二次开发边界与可验证材料

原项目提供基础的旅行规划页面、POI/天气查询和初始规划流程。本仓库的主要增量集中在：

- 使用 LangGraph 重构 Planner、Reviewer、Time Check、POI Search 等节点，增加风险门控、失败重试和用户修改后的 checkpoint 局部重规划；确定性风险门控先检查，低风险跳过 LLM Reviewer，高风险才进入审核修复；
- 将原直线距离判断替换为高德路线 API 的实际步行/驾车距离核验，距离查询失败时明确标记“未核验”，不伪造道路距离；
- 使用 SQLite + Chroma 实现结构化/语义个性化记忆，SQLite 保存事实源，Chroma 按 `user_id` 隔离语义记忆；旅行知识库按约 420 字符切块、保留约 80 字符段落重叠，默认召回 Top-3 并用 RRF 融合关键词与向量结果；
- 增加天气/POI Cache-Aside、可并行查询、FastAPI + SSE 流式进度、单次执行 Trace 和失败降级状态；
- 增加 39 条编排消融案例、真实 API 响应回放、故障轨迹测试和 GitHub Actions 离线 CI。

可验证材料：

- `docker compose up --build`：一键启动 API + Redis；
- `http://127.0.0.1:8765/docs`：FastAPI Swagger 接口文档；
- `evaluation/*.md` / `evaluation/*.json`：消融、回放、RAG、记忆和故障评测报告；
- `scripts/evaluate_risk_gate.py`、`tests/eval/ablation.py`：可复现评测入口；
- `.github/workflows/ci.yml`：GitHub Actions 编译检查、评测资产校验和 `pytest -q`。

## 项目能力

- 解析目的地、日期、预算、偏好等旅行约束，缺少关键字段时提示用户补充。
- 调用高德 POI、天气、路径等服务，生成景点候选、餐饮、酒店与出行建议。
- 使用 Planner / Reviewer 工作流生成并审查行程；候选池中没有用户指定景点时可重新检索。
- 支持用户画像与语义记忆，降低重复表达偏好的成本。
- 将明确的“少走路”和“雨天优先室内”需求抽取为结构化约束，并在 Planner、Reviewer 与离线评测中复用。
- Intent 后将天气查询与用户画像改写 fan-out 并行执行；路线的多段步行/驾车查询也并发调用，完成后再进入 Reviewer。
- 距离约束优先使用高德路线 API 的实际道路步行/驾车距离；接口失败时明确标记“未核验”，不再用直线距离冒充道路距离。
- 手动拖拽保存与路线优化同样使用道路距离：保存时并发重算相邻路段，优化时基于有向驾车距离矩阵比较候选顺序；路线服务失败时拒绝伪优化。
- 用户修改已有行程时默认复用 checkpoint 候选池；对带引号明确指定的新景点执行高德定向核验后再加入候选池。
- 前端展示规划过程、景点候选、行程路线、预算与阶段性播报。
- 提供独立的课程实验三多智能体核心：Supervisor 集中调度、Agent 私有记忆、工具权限隔离与失败重试。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| LLM / Agent | LLM Agent、LangGraph、Pydantic 结构化输出、Reviewer 工作流 |
| 记忆 / RAG | Chroma 语义记忆、SQLite 结构化记忆、关键词 + 向量混合检索 |
| 后端支撑 | Python、FastAPI、Uvicorn、Redis |
| 大模型 | OpenAI / Grok / DeepSeek / 豆包（环境变量切换） |
| 地图与工具 | 高德地图 Web 服务 API、JS API |
| 数据与记忆 | SQLite、Chroma 语义记忆（本地运行数据不提交） |
| 前端 | React/JSX、SSE、Amap JS API |
| 质量保障 | pytest、离线 Golden Set、LLM-as-Judge 评测 |

## 架构概览

```text
用户需求
  -> Intent
  -> [Query Rewrite || Weather Search]（并行后汇合）
  -> POI Search
  -> Planner Agent
  -> Route Distance Check（各路段并行）
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
LLM_PROVIDER=grok
GROK_API_KEY=你的Grok_API_Key
GROK_BASE_URL=https://api.x.ai/v1
GROK_MODEL=grok-4.6
GROK_USE_RESPONSES_API=1
GROK_REASONING_EFFORT=low
```

官方 xAI 使用 `https://api.x.ai/v1`；如果接 Grok CLI 的 OpenAI 兼容中转，把 `GROK_BASE_URL` 和 Key 按 CLI 的 `config.toml` 填入。项目不会自动读取 CLI 配置文件。

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

# 编排消融（离线结构契约，不调用外部 LLM/API）
.\.venv\Scripts\python.exe -m tests.eval.ablation --json evaluation/ablation_report.json --out evaluation/ablation_report.md

# 使用当前 .env.local 的真实 LLM/地图配置做一次安全冒烟（只输出元数据）
.\.venv\Scripts\python.exe scripts\provider_smoke.py --allow-external-calls

# 重复真实链路，输出成功率、P50/P95、Token/成本与节点耗时增量
.\.venv\Scripts\python.exe scripts\benchmark_pipeline.py --runs 3 --allow-external-calls

# 免费检查 30 用例矩阵和 30/30 已生成 fixture，不访问网络
.\.venv\Scripts\python.exe scripts\validate_eval_assets.py --require-fixtures --out evaluation\eval_asset_report.json

# 免费预估在线评测的逻辑调用和重试后供应商请求上限
.\.venv\Scripts\python.exe -m tests.eval.run_eval --dry-run

# 异常输入发现集：固定随机种子，检查 malformed 路线是否被安全拦截
.\.venv\Scripts\python.exe tests\eval\run_adversarial_risk_gate.py
```

离线测试用于验证结构化输出、调度、隔离与降级路径；涉及真实地图、天气、票价的数据必须在配置 Key 后实时调用，不能将本地测试数据当作实时事实。

消融评测固定比较三种编排：仅 Planner、Planner + 独立 Reviewer、Planner + Reviewer + Time Check。当前 39 条分层离线案例的通过数分别为 9/39、26/39、35/39，平均调用节点为 1.0、2.0、3.0；该结果用于量化硬约束节点的增益和成本，不代表线上真实 LLM 通过率。真实 API 响应回放目前为保存样本，不等同于线上压测或 SLA。

## 工程化运行与可观测性

## 平台化能力边界

- `app/llm/router.py`：已接入统一 LLM 工厂的可选模型路由。短任务（意图改写、画像/记忆抽取、短输入补充信息）可使用当前供应商的 `LLM_SMALL_MODEL`，Planner/Reviewer/Judge 保留主模型；`LLM_ROUTING_ENABLED=0` 时完全保持现有模型配置。当前只有离线路由契约，没有声称真实成本收益。
- `app/multi_agent_core/protocols.py`：本地 `AgentCard`、`MCPToolManifest`、`A2ATaskEnvelope` 契约，配合 `ToolRegistry.manifest(agent)` 和 `Supervisor.agent_cards()` 做能力发现、最小权限和审计字段演示。

这部分是可运行的协议化本地原型，不等同于已部署远程 MCP Server、A2A 网络传输或服务注册中心；旅行主链路仍按稳定依赖使用 LangGraph。

项目提供 `Dockerfile` 和 `docker-compose.yml`，可在本地同时启动 API 与 Redis：

```bash
docker compose up --build
```

健康检查和离线功能无需密钥即可启动；调用真实地图与 LLM 规划前，再复制 `.env.example` 为 `.env.local` 并填写对应 Key。

部署检查接口：

- `GET /api/health`：检查 SQLite 和 Redis 状态；
- `GET /api/metrics`：输出 HTTP 与 Agent 节点的 Prometheus 兼容指标；
- `GET /api/agent-runs/{run_id}`：登录用户查询自己的 Agent 节点耗时、并行重叠、慢节点、降级服务和单次 LLM 用量摘要；
- 所有响应包含 `X-Request-ID`，可用于关联一次 SSE 规划请求。

`/api/metrics` 还会输出 HTTP 平均/P95/最大延迟，以及 LLM 调用次数、输入/输出 token、累计 LLM 延迟、估算成本和 usage 精确率。结构化输出解析后供应商不一定保留 usage metadata：能读取时使用真实 token，读取不到时使用字符数估算并计入 `travelmind_llm_usage_exact_ratio`。成本单价不硬编码，可通过 `LLM_INPUT_USD_PER_1K` 和 `LLM_OUTPUT_USD_PER_1K` 按当前供应商账单配置。

规划 SSE 的第一帧返回 `run_id`，响应头同时包含 `X-Agent-Run-ID`。服务默认在 180 秒后终止未完成的规划，可通过 `PLAN_TIMEOUT_SECONDS` 调整。Trace 仅保留最近 200 次执行，不保存用户原始 Prompt、模型思考或模型输出。单次 Token 通过 request context 归因，仍需结合 `usage_exact_ratio` 区分供应商原始 usage 与字符估算。

Docker 默认关闭 Chroma embedding 模型下载，旅行知识库使用带来源标签的本地关键词检索，避免首次启动下载约 79 MB 模型而阻塞服务。需要完整向量检索时，将 `SEMANTIC_MEMORY_ENABLED=1`、`TRAVEL_KNOWLEDGE_ENABLED=1`；语义记忆检索仍会在 `MEMORY_LOOKUP_TIMEOUT_SECONDS`（默认 2.5 秒）后降级。

详细说明见 [`docs/production-readiness.md`](docs/production-readiness.md)。GitHub Actions 会执行 Python 编译检查和完整离线测试集，不依赖外部 API Key。

简历项目表述、可验证指标和 3 分钟演示脚本见 [`docs/interview-project-sheet.md`](docs/interview-project-sheet.md)。

在线提供商验证脚本只打印 SSE 事件数量、成功状态、错误码和耗时，不打印 API Key、Prompt 或完整行程；所有真实 LLM/地图评测入口默认拒绝执行。Planner/Reviewer 评测还要求 `--max-llm-calls`，Fixture 生成要求 `--max-api-calls`，并将结构化调用最多 3 次尝试纳入预算。第三方兼容中转的稳定性和隐私边界需按服务商条款评估。

## 说明与致谢

- 本项目用于《人工智能应用实践》课程实验和个人工程学习。
- 原始项目：[`shouzhuoshouzhuo/FloatTrip`](https://github.com/shouzhuoshouzhuo/FloatTrip)。二次开发内容应以本仓库提交记录为准。
- 请遵守高德地图、模型服务商及第三方服务的使用条款，不要提交 API Key、用户对话、用户画像或本地数据库。
