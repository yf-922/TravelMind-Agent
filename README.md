# TravelMind-Agent：LLM 多 Agent 旅行规划助手

> 基于开源项目 [FloatTrip](https://github.com/shouzhuoshouzhuo/FloatTrip) 的课程二次开发。面向“输入旅行需求，生成可解释的行程方案”的场景，重点实现多智能体协作、工具调用、个性化记忆与可复现测试。

## 二次开发边界与可验证材料

原项目提供基础的旅行规划页面、POI/天气查询和初始规划流程。本仓库的主要增量集中在：

- 使用 LangGraph 重构 Planner、Reviewer、Time Check、POI Search 等节点，增加风险门控、失败重试和用户修改后的 checkpoint 局部重规划；确定性风险门控先检查，低风险跳过 LLM Reviewer，高风险才进入审核修复；
- 将原直线距离判断替换为高德路线 API 的实际步行/驾车距离核验，距离查询失败时明确标记“未核验”，不伪造道路距离；
- 使用 SQLite + Chroma 实现结构化/语义个性化记忆，SQLite 保存事实源，Chroma 按 `user_id` 隔离语义记忆；旅行知识库按约 420 字符切块、保留约 80 字符段落重叠，默认召回 Top-3 并用 RRF 融合关键词与向量结果；
- 增加天气/POI Cache-Aside、可并行查询、FastAPI + SSE 流式进度、单次执行 Trace 和失败降级状态；
- 增加 39 条编排消融案例、真实 API 响应回放、故障轨迹测试和 GitHub Actions 离线 CI。
- `POST /api/plan/stream` 支持 `engine=supervisor`：使用登录用户范围内的角色记忆和统一 SSE 事件；行程修改与确认会沿用同一编排引擎，硬约束复核优先于模型审核结论。

可验证材料：

- `docker compose up --build`：一键启动 API + Redis；
- `http://127.0.0.1:8765/docs`：FastAPI Swagger 接口文档；
- `evaluation/*.md` / `evaluation/*.json`：消融、回放、RAG、记忆和故障评测报告；
- `scripts/evaluate_risk_gate.py`、`tests/eval/ablation.py`：可复现评测入口；
- `.github/workflows/ci.yml`：GitHub Actions 编译检查、评测资产校验和 `pytest -q`。

## 项目能力

### 独立旅游知识检索评测（未切换生产默认）

`knowledge/benchmark_v1` 保存公开官方页面快照、稳定 chunk ID、URL、城市、主题、采集日期、内容哈希、证据组和采集失败记录。
当前只有 38 个有效快照 chunk（上海、景德镇），未达到六城 120–180 chunk 目标；106 条问题为程序起草且全部 `pending_review`，不称人工 gold。
新评测入口 `scripts/sweep_rag_v2.py` 对比关键词、固定 jieba 分词的 BM25、真实 Chroma 向量、BM25+向量标准 RRF、关键词+向量标准 RRF以及旧取整 RRF。
索引位于独立 `data/travel_knowledge_benchmark_v1`，按语料指纹命名 collection，生产语料、索引、默认 Top-3 保持不变。
显式向量/融合失败会报告错误，不能用关键词降级冒充融合；开发扫 k=1/2/3/5/8/10 和 c=20/60/100。
仅待审核开发草稿执行烟测，冻结测试未执行；详细边界与命令见 [检索评测报告](docs/rag_retrieval_v2_smoke.md)。
安装独立依赖：`python -m pip install -r requirements-rag-eval.txt`；生成端及 Judge 尚未运行，预算预估见报告。

### 有界候选池 ReAct

主链路在需求分析、查询改写和天气查询完成后，先通过 `attraction_search`
建立不依赖 LLM 的城市基础候选池，再进入
`candidate_react -> candidate_search -> candidate_validator` 补检索循环。
模型只提出最多两个类别/景点搜索动作；工具执行高德查询、合并去重和来源标注，
校验器检查数量、明确景点覆盖、室内外约束及类型多样性。
最多执行两轮，每次请求的基础与补充检索共享四次 HTTP 尝试预算（含重试），Redis 命中不计外部请求。
这个预算目前仅覆盖候选池检索，不包含天气、餐馆和道路查询。
LLM 失败或对空池提出停止时丢弃模型决策，使用服务端预定义类别查询；已有池则保留结果。
API 失败或预算耗尽时保留已有候选并返回 `insufficient`，
Planner 接收覆盖缺口，最终结果包含 `candidate_pool` 状态与统计。
室内外标签来自高德类型/名称启发式，未知标签不计为已满足；明确用户限制优先于多样性。
有评分的候选执行最低评分过滤，无评分但坐标有效的真实 POI 保留并标注 `rating_status=unknown`。
配置通过 `TravelPlanState` 的 `candidate_min_per_day`、`candidate_max_rounds`、
`candidate_api_budget` 控制；默认数量目标为天数乘每日两个景点，再加至少两个备选，受 `max_spots` 限制。

离线验收：`python -m pytest tests/test_candidate_react.py -q`。
六场景契约评测：`python scripts/evaluate_candidate_react.py`，打印 JSON，
包含覆盖数、明确景点命中、室内外分布、检索轮数、请求次数和不足场景；不调用真实 LLM/API。
这些结果用于验证循环与预算机制，实际检索质量仍需要独立真实 Provider 对照。

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

### Supervisor 协作链路验证

前端可选“协作规划”，对应 `engine=supervisor`；默认仍为 LangGraph。
Planner、Reviewer、Time Check 分别读取会话内的角色私有记忆，SQLite 事务快照
和 4,000 字符预算限制跨请求干扰与上下文膨胀。公共状态保存候选池、当前路线、
审核结论和必要反馈，不保存角色私有对话。未通过审核的草稿不会作为成功行程展示。
修改景点类型等偏好时，Supervisor 会先判断旧候选池是否覆盖；不足时最多执行三次
关键词补查，只合并高德返回的真实 POI，补查失败会进入审核失败而不是编造满足。
首轮规划还会查询餐馆候选并由联合 Planner 生成餐位时间块；餐馆候选、营业时间、
道路距离和景点重叠会进入同一审核链路。仅修改时间时复用餐馆池，修改饮食偏好时重新查询。
初始化或运行中的私有记忆读写失败均会在结果中标记 `private_memory` 降级，
本次规划可继续，但不声称记忆已持久化。四进程 SQLite 测试验证同会话 80 条
并发写入无丢失及用户/角色隔离；这不是多实例 API 负载验证。
Chroma 语义检索也返回 `ok/skipped/degraded` 状态；不可用时仍可规划，但最终
结果会明确列出 `semantic_memory` 降级，不把空召回误报为“没有用户偏好”。
用户可通过 `DELETE /api/profile/semantic-memory` 删除自己的长期语义记忆，
删除失败返回 503，不会伪造成功。

```bash
python scripts/evaluate_supervisor_parity.py --dry-run
python scripts/evaluate_supervisor_api.py --dry-run
node --test scripts/test_sse_client.cjs
```

真实调用必须显式传入 `--allow-external-calls` 和相应预算；API smoke 使用临时
SQLite，关闭 Chroma 与后台画像提取，仍执行真实天气、POI、模型及行程补充节点。
预算为保守调用上界估算，不是供应商计费硬限额。验证边界和尚未完成的验收见
[`docs/supervisor-runtime-progress.md`](docs/supervisor-runtime-progress.md)。

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

规划 SSE 的第一帧返回 `run_id`，响应头同时包含 `X-Agent-Run-ID`。默认不设整单 180 秒截止，而按节点计时：Planner 150 秒、intent 90 秒、查询改写 60 秒、天气 15 秒、ReAct 决策 60 秒，其余节点 120 秒。可通过 `NODE_TIMEOUT_SECONDS` 设置通用上限，或通过 `NODE_TIMEOUT_PLANNER_SECONDS` 等设置单节点上限。并行节点独立计时；其他分支进度和心跳不会刷新卡住节点的期限。超过期限返回 `NODE_TIMEOUT` 和节点名，不将未完成结果当成功；取消协程不能强制停止已进入线程的同步调用，底层 HTTP 仍保留请求超时。有需要时运营可显式设置 `PLAN_TIMEOUT_SECONDS` 作为整单应急上限（默认 0，关闭）。Trace 仅保留最近 200 次执行，不保存用户原始 Prompt、模型思考或模型输出。单次 Token 通过 request context 归因，仍需结合 `usage_exact_ratio` 区分供应商原始 usage 与字符估算。

Docker 默认关闭 Chroma embedding 模型下载，旅行知识库使用带来源标签的本地关键词检索，避免首次启动下载约 79 MB 模型而阻塞服务。需要完整向量检索时，将 `SEMANTIC_MEMORY_ENABLED=1`、`TRAVEL_KNOWLEDGE_ENABLED=1`；语义记忆检索仍会在 `MEMORY_LOOKUP_TIMEOUT_SECONDS`（默认 2.5 秒）后降级。

详细说明见 [`docs/production-readiness.md`](docs/production-readiness.md)。GitHub Actions 会执行 Python 编译检查和完整离线测试集，不依赖外部 API Key。

简历项目表述、可验证指标和 3 分钟演示脚本见 [`docs/interview-project-sheet.md`](docs/interview-project-sheet.md)。

在线提供商验证脚本只打印 SSE 事件数量、成功状态、错误码和耗时，不打印 API Key、Prompt 或完整行程；所有真实 LLM/地图评测入口默认拒绝执行。Planner/Reviewer 评测还要求 `--max-llm-calls`，Fixture 生成要求 `--max-api-calls`，并将结构化调用最多 3 次尝试纳入预算。第三方兼容中转的稳定性和隐私边界需按服务商条款评估。

## 说明与致谢

- 本项目用于《人工智能应用实践》课程实验和个人工程学习。
- 原始项目：[`shouzhuoshouzhuo/FloatTrip`](https://github.com/shouzhuoshouzhuo/FloatTrip)。二次开发内容应以本仓库提交记录为准。
- 请遵守高德地图、模型服务商及第三方服务的使用条款，不要提交 API Key、用户对话、用户画像或本地数据库。
