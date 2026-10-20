<div align="center">

# TravelMind · 途见

### 把目的地交给我，你只管期待出发。

**支持长期记忆与动态修改的 LLM 多 Agent 旅行规划应用**

[![Python CI](https://github.com/yf-922/TravelMind-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yf-922/TravelMind-Agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Chroma](https://img.shields.io/badge/Memory-Chroma%20%2B%20SQLite-B74724)
![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white)

自然语言需求 → 真实景点候选 → 景点与餐饮联合规划 → 道路核验与审核 → 可修改的每日行程

[产品预览](#产品预览) · [核心设计](#核心设计) · [快速开始](#快速开始) · [评测与结果](#评测与结果) · [压测数据](#本机-http-并发压测) · [源码导航](#源码导航)

</div>

![TravelMind 旅行规划首页](static/images/规划页.png)

## 项目简介

TravelMind 将一句旅行需求转换为包含景点、餐饮、天气、交通与时间安排的行程。用户可以继续提出“第二天换个景点”“少走一点路”等修改意见，系统复用已保存的规划状态，再生成并检查新路线。

项目围绕旅行 Agent 的三个实际问题展开：**模型可能编造地点，生成的时间表未必可执行，修改需求容易重复查询**。地点事实来自外部工具，路线由 Planner 生成，规则与 Reviewer 分别检查可计算约束和语义质量，LangGraph 负责条件路由与有限次返工。

| 用户遇到的问题 | 项目的处理方式 |
| :--- | :--- |
| 推荐了不存在或不匹配的景点 | 先建立高德 POI 候选池，再约束 Planner 从候选中选择 |
| 到达时未开放、时间冲突、步行过长 | 规则检查时间结构，使用高德实际步行 / 驾车距离与耗时核验 |
| 形式正确，但偏好不匹配或节奏过密 | LLM Reviewer 读取用户需求、候选事实和路线，给出修改意见 |
| 修改一个景点也要重新查询所有信息 | 保存应用层 checkpoint，复用需求、天气与候选，必要时定向补查 |
| 等待时不知道系统在做什么 | SSE 返回节点进度、阶段摘要、最终结果及失败原因 |
| 每次都要重复说明喜好 | SQLite 保存结构化画像，Chroma 按用户隔离召回语义记忆 |

## 产品预览

**一份行程，一张地图，一条可继续修改的时间线。**

![每日行程、天气、候选景点和高德地图](static/images/规划详情.png)

<details>
<summary><strong>展开查看：流式规划过程与用户画像</strong></summary>

**节点进度**：显示意图分析、查询、规划与审核过程。以下为仓库保存的界面截图，具体节点及轮次以当前执行为准。

![规划阶段展示](static/images/规划进程.png)

**用户画像**：查看与编辑景点、饮食和出行习惯偏好。

![用户画像界面](static/images/我的画像.png)

</details>

可以从这样的请求开始：

> 南京明天出发，玩三天，喜欢历史和博物馆，想吃当地小吃，每天别安排太满，下雨优先室内。

再提出修改：

> 第二天把一个景点换成美术馆，其他安排尽量保留。

建议先注册并登录，完整体验行程保存、画像更新和后续修改。

## 核心设计

### 01 · 用状态图管理规划过程

主链路使用 **LangGraph + Pydantic State + 条件边**。模型输出意图、行程或审核结论等结构化业务结果；代码更新状态并选择下一步，模型不能直接指定任意节点或跳过校验。

```mermaid
flowchart TD
    U[自然语言旅行需求] --> I[Intent：目的地 / 日期 / 偏好]
    I -->|信息缺失| C[Clarify：请求补充信息]
    I -->|信息完整| Q[Query Rewrite：结合画像改写]
    I -->|信息完整| W[Weather Search：天气查询]
    I -->|信息完整| M[Main Meal Search：餐馆候选]
    Q --> A[Attraction Search：基础景点候选]
    W --> A
    A --> R[Candidate ReAct：选择补查动作]
    R --> S[Candidate Search：执行工具 / 校验 / 去重]
    S --> V{Candidate Validator：覆盖检查}
    V -->|有缺口且预算未耗尽| R
    V -->|覆盖满足或达到上限| J[Candidate Ready：保留覆盖状态]
    J --> P[Joint Planner：景点 + 餐饮时间块]
    M --> P
    K[候选关联的旅行知识] -.->|本轮参考证据| P
    P --> D[Route Distance Check：并发核验道路]
    D --> G{Risk Gate：确定性风险检查}
    G -->|需要语义审核| L[LLM Reviewer]
    G -->|仅需时间核查| T[Time Check]
    G -->|满足跳过条件| O[餐饮结果整理 + 景点贴士]
    L -->|驳回且未达上限| P
    L -->|通过或达到上限| T
    T -->|需要修改且未达上限| P
    T -->|需补充语义复核| L
    T -->|检查结束| O
    O --> F[Finalize：最终校验 / 结果 / 失败原因]
    F --> E[SSE 展示与行程保存]

    classDef model fill:#fff0e7,stroke:#bc4b24,color:#512b20;
    classDef tool fill:#edf5f0,stroke:#557963,color:#233c2e;
    classDef gate fill:#fff8df,stroke:#b08a2e,color:#58491f;
    class I,Q,R,P,L,T model;
    class W,M,A,S,D tool;
    class V,G,F gate;
```

查询改写、天气与餐馆候选依赖 Intent，可以并行启动；基础景点检索等待查询改写与天气，联合 Planner 等待景点池和餐馆池。路线核验中的独立路段也并发执行。图中“餐饮结果整理 + 景点贴士”合并展示两个并行分支，Finalize 等待两者完成。

**两种运行方式**：默认 `engine=langgraph` 使用共享 `TravelPlanState`；可选 `engine=supervisor` 使用代码主管调度、角色私有记忆和工具权限控制。两条链路共用部分生产节点，主图不依赖一个自由决定节点的 LLM Supervisor。详见 [Supervisor 运行时说明](docs/supervisor-runtime-progress.md)。

### 02 · 候选池先核验，ReAct 只补缺口

先通过城市级查询建立基础候选池，再用 `candidate_react → candidate_search → candidate_validator` 进行有界补查。模型只返回允许的搜索动作，例如室内景点或用户点名地点；实际 POI 来自高德响应。

| 保护机制 | 当前实现 |
| :--- | :--- |
| 动作范围 | Pydantic 枚举约束搜索类型，每轮最多 2 个动作 |
| 点名景点 | 查询必须来自用户明确点名；返回 POI 名称严格匹配 |
| 候选写入 | 检查名称、经纬度合法性；普通候选按评分过滤；按名称去重 |
| 重复查询 | 城市、关键词和类型生成 `query_key`，已执行查询不再调用 |
| 终止条件 | 补查最多 2 轮；基础与补查共享默认 4 次候选搜索 HTTP 尝试预算，包含重试 |
| 不足与失败 | 保留已有候选及缺口，标记 `insufficient`，交给后续规划与校验 |

缓存命中不消耗外部调用预算；上述预算仅覆盖景点候选搜索，不包括天气、餐馆和道路查询。覆盖检查涉及数量、点名命中及室内外分布；室内外标签来自类型 / 名称启发式，未知不算已满足。

代码入口：[candidate_react.py](app/planning/candidate_react.py) · [POI 适配器](app/providers/amap/poi.py)

### 03 · 硬约束与软约束分工审核

**规则处理能明确计算的问题，模型补充需要理解语义的问题。**

| 层次 | 检查内容 | 处理方式 |
| :--- | :--- | :--- |
| 规则与道路工具 | 候选越界、重复景点、时间结构、开放时间冲突、道路耗时与步行限制、餐位冲突 | 写入风险标记，供路由、返工与最终检查使用 |
| LLM Reviewer | 当前兴趣、排除偏好、游玩节奏及整体合理性 | 输出通过位、问题与修改意见；未通过时有限次回退 Planner |
| 时间核查 | 日程时间与开放窗口，需要修改时再复核 | 独立修正循环，结束后进入结果整理 |
| Finalize | 检查当前状态并整理结果 | 返回成功状态、降级项或无法完成原因 |

当前门控根据规则风险及用户偏好决定是否进入 Reviewer：**有兴趣 / 习惯要求或明确偏好表述，也会触发语义审核**，并非“硬约束没有风险就必然跳过”。主链路默认 `max_review_rounds=3`，时间核查也有独立上限。

审核发现路线问题时，主图回到 Planner 重排；审核阶段自动回到候选检索不是当前主图已实现的通用分支。候选补查主要发生在初始覆盖检查，以及修改请求中的定向核验。路线在同一候选池上仍无法满足时，由后续校验返回失败原因。

代码入口：[风险规则、Reviewer 与 Finalize](app/planning/nodes.py) · [结构化 State 与输出](app/planning/schemas.py)

### 04 · 保存状态，让修改从已有信息继续

登录用户的行程记录保存 `plan_json` 与 `planner_state_json`。修改时恢复目的地、日期、偏好、天气、景点和餐馆候选等上下文，运行修改子图，再做道路与审核检查。用户从搜索结果选择新景点时，服务端核验后才合并进入候选池。

这里的 **checkpoint 是应用保存到 SQLite 的规划状态快照**，用于行程修改与信息复用；主 LangGraph 尚未接入支持任意节点恢复的持久化 checkpointer。

Planner 重新写入当前路线；知识来源按轮覆盖，空召回不沿用旧证据；存在修改顾虑时，返回确认事件并保存待确认状态，用户确认后继续。

代码入口：[规划 API 与 checkpoint 保存](app/main.py) · [修改子图](app/planning/graph.py) · [行程存储](app/core/memory.py)

### 05 · 长期记忆与公共知识各司其职

| 存储 | 保存什么 | 用途 |
| :--- | :--- | :--- |
| SQLite · `data/app.db` | 用户账户、景点 / 饮食 / 习惯画像、历史行程、规划快照、待确认修改 | 精确读取、更新与持久化 |
| Chroma · `data/semantic_memory/` | 从用户需求提取的自然语言记忆及 `user_id` 元数据 | 语义召回；查询与删除均按用户过滤 |
| 旅行知识 · `knowledge/travel/` | 公共指引和带来源的具名景点事实卡 | 给 Planner 比较已核验候选提供参考 |
| Redis | 高频 POI 与天气查询响应 | Cache-Aside 缓存；POI TTL 12 小时，天气 TTL 4 小时 |

用户本次明确要求优先于历史偏好。主 LangGraph 的节点通过共享 State 交接信息；角色私有对话记忆属于可选 Supervisor 链路。Chroma 语义用户记忆与公共知识混合检索是两种能力。

知识检索按城市和候选实体过滤，将来源、URL 与采集日期交给 Planner；知识只能用于候选比较，不能新增候选池外地点。票价、余量与临时政策不能由历史快照当作当前事实。

检索实验独立比较关键词、BM25、向量及 RRF 混合召回，使用固定 revision 的中文 BGE；**生产 Chroma 默认 embedding 与该评测配置分开管理**。实验兴趣知识默认关闭，详见 [RAG 评测说明](docs/rag_benchmark_v2.md) 与 [最终选点对照方案](docs/route_interest_evaluation.md)。

## 快速开始

### 方式一：Docker Compose

需要 Git、Docker 与支持 `env_file.required` 的 Docker Compose v2.24+。

```bash
# 获取项目
git clone https://github.com/yf-922/TravelMind-Agent.git
cd TravelMind-Agent

# 首次配置：复制模板，并填写地图与模型 Key
cp .env.example .env.local

# 启动 FastAPI 与 Redis
docker compose up --build -d

# 查看应用日志
docker compose logs -f travelmind
```

Windows PowerShell 复制模板使用 `Copy-Item .env.example .env.local`。

| 入口 | 地址 |
| :--- | :--- |
| Web 应用 | http://127.0.0.1:8765 |
| Swagger 接口文档 | http://127.0.0.1:8765/docs |
| 依赖健康检查 | http://127.0.0.1:8765/api/health |
| Prometheus 格式指标 | http://127.0.0.1:8765/api/metrics |

Compose 通过命名卷保留应用与 Redis 数据。未配置 Key 也可以启动页面与健康检查，真实旅行规划需要高德 Web 服务 Key 和一个可用模型提供商。

**Docker 默认关闭 Chroma 能力**，避免首次启动下载 embedding 模型。开启时需同时修改 `docker-compose.yml` 中的 `SEMANTIC_MEMORY_ENABLED` 和 `TRAVEL_KNOWLEDGE_ENABLED`；仅修改 `.env.local` 不会覆盖 Compose 中的同名设置。

### 方式二：本地 Python

推荐 Python 3.11 或 3.12。前端由应用直接提供，基本运行无需另行执行 npm 构建。

```powershell
# Windows PowerShell
git clone https://github.com/yf-922/TravelMind-Agent.git
cd TravelMind-Agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env.local

# 编辑 .env.local 后启动
python run.py
```

<details>
<summary><strong>Linux / macOS 启动命令</strong></summary>

```bash
git clone https://github.com/yf-922/TravelMind-Agent.git
cd TravelMind-Agent
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env.local
# 填写环境变量后启动
python run.py
```

</details>

### 最小配置与可选能力

下面使用 DeepSeek 提供商；模型名称请与账户当前可访问的模型保持一致。

```dotenv
# 高德 Web 服务 Key：后端 POI、天气与道路查询
AMAP_API_KEY=your_amap_web_service_key

# 高德 JS API Key 与安全密钥：前端地图展示
AMAP_JS_KEY=your_amap_js_key
AMAP_JS_SECURITY_CODE=your_amap_js_security_code

# 选择一个模型提供商；支持 deepseek / openai / grok / doubao
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
# DEEPSEEK_MODEL=your_available_model

# 本地 Redis 可选；不使用缓存时留空
REDIS_URL=

# 普通运行按节点超时；关闭整单 180 秒截止
# 模板原值为 180，复制后请按需要修改
PLAN_TIMEOUT_SECONDS=0

# 先运行基础链路，再按需开启语义能力
SEMANTIC_MEMORY_ENABLED=0
TRAVEL_KNOWLEDGE_ENABLED=0
TRAVEL_INTEREST_KNOWLEDGE_ENABLED=0
```

其他提供商的字段见 [.env.example](.env.example) 和 [LLM 接入说明](LLM_PROVIDERS.md)。环境变量、数据库、运行时索引及账本在 `data/` 或 `.env.local` 中，不提交到仓库。

## 流式交互与可观测性

SSE 负责**服务端到前端**的进度推送，Agent 之间通过 State 或结构化消息传递信息。规划第一帧包含 `run_id`，响应头包含 `X-Request-ID` 与 `X-Agent-Run-ID`。

| 事件 / 接口 | 作用 |
| :--- | :--- |
| `run` / `stage` / `stage_summary` | 标识执行、当前节点及阶段摘要 |
| `modification_warning` | 返回修改顾虑，等待用户确认 |
| `result` | 返回最终成功状态、路线或失败信息 |
| `GET /api/agent-runs/{run_id}` | 登录用户查看自己的节点耗时、降级项和 LLM 用量摘要 |
| `GET /api/metrics` | HTTP 时延、节点统计、LLM 调用、输入 / 输出 Token 与 usage 精确率 |

默认节点期限：Planner 150 秒、Intent 90 秒、查询改写与 ReAct 决策 60 秒、天气 15 秒，其余 120 秒。可用 `NODE_TIMEOUT_SECONDS` 或 `NODE_TIMEOUT_PLANNER_SECONDS` 等覆盖。整单应急上限由 `PLAN_TIMEOUT_SECONDS` 控制；代码默认关闭，但环境模板填写了 180 秒，部署时需显式选择。

Token 统计区分供应商实际 usage 与字符估算；只有配置单价后成本估算才有意义。执行 Trace 保留最近 200 次，位于进程内存，不保存原始 Prompt、模型思考与完整回复。SSE 尚不提供基于 event ID 的断线重放。

<details>
<summary><strong>接口调用示例：注册并发起流式规划</strong></summary>

以下为 Bash 示例，依赖 `curl` 和 Python。注册用户名应使用未占用的名称，密码自行设置。

```bash
# 注册返回 token；已有账号可改用 /api/auth/login
TRAVELMIND_TOKEN=$(curl -fsS http://127.0.0.1:8765/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"travelmind-demo","password":"replace-with-your-password"}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["token"])')

# -N 禁止 curl 缓冲，实时显示 SSE 数据
curl -N http://127.0.0.1:8765/api/plan/stream \
  -H "Authorization: Bearer $TRAVELMIND_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"南京明天出发三日游，喜欢历史和博物馆，慢节奏","engine":"langgraph","max_review_rounds":3}'
```

修改请求在相同接口传入 `plan_id` 与 `modification_notes`。个人画像、历史行程、景点搜索与路线编辑的完整 schema 可在本地 Swagger 查看。

</details>

## 评测与结果

### 已复现的工程验证

2026-10-20，本地 Python 3.12.8 全量回归：**501 项 pytest 通过**（42.83 秒，3 条依赖弃用警告），另有 **5 项 SSE 客户端测试通过**；Python 编译与三项评测资产校验均完成。测试覆盖结构化输出、调度、候选边界、记忆隔离、修改链路、风险检查、SSE 结果处理与降级契约。GitHub Actions 配置同类检查，具体远程执行状态以上方 CI 徽章为准。

### 本机 HTTP 并发压测

**2026-10-20 实测 8,100 次请求**：真实 TCP 请求进入独立 Uvicorn 单进程，经 FastAPI 和临时 SQLite 返回。使用 50 个合成用户、1,000 条历史行程；画像和历史接口同时检查返回数据归属，健康检查验证数据库状态。测试不读取个人数据库，不调用模型或地图。

三个接口分别测试并发 **1 / 8 / 32**，每档 **300 请求 × 3 次重复**，每次另预热 12 请求。以下展示并发 8 的合并结果，P95 及吞吐均包含失败请求；本轮未出现失败。

| 接口 | 请求数 | 吞吐 req/s | P50 | P95 | 失败 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 健康检查 `/api/health` | 900 | 1,595.74 | 3.818 ms | 10.463 ms | 0 |
| 用户画像 `/api/profile` | 900 | 1,125.28 | 5.271 ms | 17.285 ms | 0 |
| 历史行程 `/api/history` | 900 | 797.94 | 9.700 ms | 14.020 ms | 0 |

**并发增加不等于吞吐持续增加**：并发 32 时，画像接口 P95 升至 **105.489 ms**，吞吐降至 **796.46 req/s**；历史接口 P95 升至 **62.811 ms**。本轮结果支持继续排查高并发排队、数据库与连接开销，尚未定位具体瓶颈。

这是同机客户端的闭环读接口测试，不包含 LLM 规划、SSE 长连接、外部 API、持续写入或多实例部署，不能据此估算旅行规划容量。完整 9 档结果、三次重复、逐请求耗时、错误和源码指纹见 [HTTP 压测报告](evaluation/http_load_20261020.md) / [原始 JSON](evaluation/http_load_20261020.json)。

### 规划节点并行基准

使用生产状态图 / 道路核验节点，将外部操作替换为固定延迟的离线实现；与相同工作量的串行执行比较，每项重复 **30 次**。该实验用于验证调度效果。

| 对照 | 固定工作量 | 串行 P50 | 并行 P50 | P50 加速比 |
| :--- | :--- | ---: | ---: | ---: |
| 查询改写、天气、餐馆候选 | 3 分支，各 40 ms | 132.684 ms | 50.707 ms | **2.62×** |
| 餐饮结果、景点贴士 | 2 分支，各 40 ms | 88.900 ms | 48.519 ms | **1.83×** |
| 道路距离核验 | 8 路段，各 20 ms | 164.091 ms | 24.186 ms | **6.79×** |

checkpoint 另有阶段成本模拟：完整重跑 P50 **94.741 ms**，复用上游阶段后 **64.591 ms**，模拟减少 **31.8%**；固定输入预算假设由 9,900 减至 8,400。该项未执行真实模型，也不是历史版本实测，不能当作真实 Token 节省或线上耗时收益。

完整 P50 / P95、逐次配对数据、环境与源码指纹见 [工程基准报告](evaluation/offline_engineering_benchmark.md) / [原始 JSON](evaluation/offline_engineering_benchmark.json)。真实模型的质量与用量另见下面的配对消融。

### 真实模型配对消融

最新 Reviewer 消融使用 **60 个自建场景、3 组、180 个记录结果**：30 个任务场景与 30 个鲁棒性场景。共享真实模型生成的首轮联合 Planner 草案，使用相同候选与回放；规则组和 Reviewer 组均允许最多 3 次返工，并移除 Time Check 以隔离 Reviewer 作用。

| 指标 | 仅 Planner | 规则 + 返工 | 规则 + Reviewer + 返工 |
| :--- | ---: | ---: | ---: |
| 任务成功 | 26/30 | 30/30 | 30/30 |
| 任务错误放行 | 4/30 | 0/30 | 0/30 |
| 鲁棒性成功或明确拒绝 | 14/30 | 18/30 | 20/30 |
| 实际恢复 | 14/30 | 15/30 | 15/30 |
| 有原因的安全拒绝 | 0/30 | 3/30 | 5/30 |

**结论**：完整审核闭环较仅规划的任务成功率提高 **13.3 个百分点**，本批提升主要来自规则与返工；Reviewer 没有进一步提高普通任务通过数。它在“必须安排不存在景点”的请求中识别出无解条件，拒绝把替代路线当作满足请求。

这是实际生产规划 / 审核节点的受控实验：上游候选与道路响应使用固定回放，评分为预设代码 rubric，不能当作人工审查结论或完整线上服务成功率。Reviewer 组仍有 9/30 鲁棒性错误放行，道路证据缺失是下一项重点。

<details>
<summary><strong>查看资源开销与复现边界</strong></summary>

实验使用 `grok-4.6`，实际付费尝试 196 次，记录输入 428,869、输出 195,661 Token；超时调用用量未知，费用为 `unpriced`。各组资源归因将共享草案计入每组逻辑开销，不能相加当作实际账单。

| 组别 | 记录的逻辑 Token | 平均实测耗时 |
| :--- | ---: | ---: |
| 仅 Planner | 276,655 | 36.3 秒 |
| 规则 + 返工 | 324,014 | 42.6 秒 |
| 规则 + Reviewer + 返工 | 573,825 | 69.9 秒 |

耗时包含天气变化 / 修改场景的共享前置草案；一条未恢复耗时的共享失败不参与均值。详细实验与逐项失败解释见 [Reviewer 消融报告](docs/reviewer_ablation_v2.md)。旧版 [任务与鲁棒性报告](docs/task_robustness_real_v1.md) 保留评分资产缺陷和原始失败，不覆盖为新版结果。

</details>

### RAG：分别检验召回与最终选点

| 评测 | 资产 / 结果 | 当前状态 |
| :--- | :--- | :--- |
| 旧检索契约集 | 30 条查询：24 条可回答、6 条无答案；关键词与混合均命中 24/24，Top-1 分别 24/24、23/24 | 未显示混合检索优势；旧 `Recall@3` 实为来源 Hit@3 的别名 |
| 六城检索评测 v2 | 128 个语料块、120 条草稿查询，开发 / 测试各 60；关键词、BM25、向量及 3 种融合对照 | 工具与资产就绪；正式报告要求人工标注、来源确认和冻结 |
| 最终路线选点对照 | 南京、上海、三亚 30 个案例，相同候选池对比有 / 无知识 | 用最终选点而非说明文字衡量收益，正式结论仍需冻结评测 |

128 块含 **28 块项目规划建议**，不全部是官方事实。v2 区分 Recall、Hit、MRR、nDCG、Precision、事实覆盖、失败率与可用率；融合失败不会偷偷降级为关键词成功。BGE 固定 revision、语料和配置指纹参与索引及冻结验证。

目前检索增强仍保留实验开关，尚未完成支持其生产收益的正式冻结报告。混合检索是否优于单路，需要由相同数据与明确标注的实验结果决定。

详见 [早期检索报告](docs/rag_test_report.md) · [v2 协议](docs/rag_benchmark_v2.md) · [实施记录](docs/rag_benchmark_v2_implementation_report.md) · [最终路线对照](docs/route_interest_evaluation.md)。

## 测试与复现

以下检查不需要真实 LLM 或高德调用。

```bash
# 安装测试与检索评测依赖
python -m pip install -r requirements-dev.txt -r requirements-rag-eval.txt

# 与 CI 一致的基本检查
python -m compileall -q app tests scripts
python scripts/validate_eval_assets.py
python scripts/validate_rag_assets.py --corpus knowledge/benchmark_v2_curated/chunks.json --cases evaluation/rag_benchmark_v2_draft.json --require-quota
python scripts/review_route_interest.py validate
python -m pytest -q
node --test scripts/test_sse_client.cjs

# 本机 HTTP 压测：独立服务进程与临时 SQLite，约 8,100 次实测请求
python scripts/benchmark_http.py --requests 300 --concurrency 1 8 32 --warmups 12 --repetitions 3

# 调度基准：固定外部延迟，每组 30 次，输出 JSON 与 Markdown
python scripts/offline_engineering_benchmark.py --runs 30 --route-legs 8 --fanout-delay-ms 40 --provider-delay-ms 20 --stage-unit-delay-ms 2

# 有界候选检索的六场景契约检查
python scripts/evaluate_candidate_react.py

# 独立 Supervisor 演示：离线工具，不需要 API Key
python -m app.multi_agent_core.demo_run --offline --city Beijing --request "Plan a relaxed cultural day trip"
```

<details>
<summary><strong>展开查看：离线消融、检索开发实验与真实评测入口</strong></summary>

**历史结构消融**：39 条人工构造案例，9/39 → 26/39 → 35/39 是离线结构契约成绩，不是最近真实模型消融的结果。

```bash
python -m tests.eval.ablation --json evaluation/ablation_report.json --out evaluation/ablation_report.md
```

**真实节点评测预检**：不加 `--execute` 时不调用真实模型，生成运行记录到被忽略的 `data/`。

```bash
python scripts/evaluate_task_robustness.py --out data/eval/task_robustness_offline.json
python scripts/evaluate_reviewer_ablation.py
```

**RAG 开发实验**：先显式下载固定 BGE，再运行开发集。该命令使用待审标签，只产生探索性报告；模型下载需要网络和本地存储，查询阶段不会下载模型。

```bash
python scripts/prepare_rag_embedding.py --download
python scripts/sweep_rag_v2.py --cases evaluation/rag_benchmark_v2_draft.json --corpus knowledge/benchmark_v2_curated/chunks.json --out data/rag_v2_dev_draft.json
```

人工审核与冻结命令见 [v2 协议](docs/rag_benchmark_v2.md)。真实模型 / 地图运行须使用相应入口的显式授权开关及调用上限，执行前查看其 `--help`；缺失实际 Token、费用或执行结果的字段保留未知。原始运行账本、逐请求输出与本地索引保存在 `data/`，不提交。

</details>

## 源码导航

| 层次 | 技术 / 设计 | 入口 |
| :--- | :--- | :--- |
| Agent 编排 | LangGraph、条件边、有限循环、Pydantic 结构化输出 | [graph.py](app/planning/graph.py)、[schemas.py](app/planning/schemas.py) |
| 规划与审核 | 联合 Planner、规则门控、Reviewer、Time Check | [nodes.py](app/planning/nodes.py) |
| 工具检索 | 高德 POI、天气、步行 / 驾车路径 | [providers](app/providers)、[candidate_react.py](app/planning/candidate_react.py) |
| 用户记忆 | SQLite 画像 / 行程、Chroma 用户过滤 | [memory.py](app/core/memory.py)、[semantic_memory.py](app/core/semantic_memory.py) |
| 公共知识 / RAG | 候选实体过滤、关键词 / 向量、独立 BM25 + BGE 评测 | [travel_knowledge.py](app/core/travel_knowledge.py)、[rag_retrieval.py](app/evaluation/rag_retrieval.py) |
| 服务与交互 | FastAPI、SSE、React / JSX、高德 JS API | [main.py](app/main.py)、[frontend](frontend) |
| 可靠性与观测 | Redis、节点超时、Trace、Token / 延迟指标 | [agent_runs.py](app/core/agent_runs.py)、[cache.py](app/core/cache.py) |
| 实验协作引擎 | Supervisor、私有角色记忆、工具白名单、本地消息契约 | [multi_agent_core](app/multi_agent_core) |
| 部署与验证 | Docker Compose、pytest、GitHub Actions | [Compose](docker-compose.yml)、[CI](.github/workflows/ci.yml) |

```text
TravelMind-Agent/
├── app/
│   ├── planning/           # 主状态图、候选 ReAct、联合规划与审核
│   ├── multi_agent_core/   # 可选 Supervisor 与私有记忆协作引擎
│   ├── providers/          # 地图、天气、票务等外部适配器
│   ├── core/               # 用户记忆、缓存、鉴权、指标与执行 Trace
│   ├── evaluation/         # 检索协议、路线评分与故障评测
│   ├── llm/                # 提供商适配、结构化调用与模型路由
│   └── api/                # 用户、画像、历史与路线编辑接口
├── frontend/               # 页面、地图、时间线与 SSE 客户端
├── static/images/          # 项目界面截图
├── knowledge/              # 旅行知识、来源快照与版本化评测语料
├── evaluation/             # 版本化评测资产和可公开报告
├── tests/                  # 单测、集成测试、契约及评测 Fixture
├── scripts/                # 采集、校验、评测、账本与报告工具
├── docs/                   # 架构、实验设计和进展说明
└── data/                   # 本地数据库 / 索引 / 账本；Git 忽略
```

## 常见问题

<details>
<summary><strong>页面能打开，但规划不能运行？</strong></summary>

检查 `.env.local` 中的 `LLM_PROVIDER` 与对应 Key 是否匹配，高德 Key 是否为 Web 服务类型，模型名是否可用。地图展示另需 JS API Key 与安全密钥；它们与后端 Web 服务 Key 不通用。查看 `docker compose logs -f travelmind` 或本地终端的错误码。

</details>

<details>
<summary><strong>规划超过 180 秒就停止？</strong></summary>

`.env.example` 中填写了 `PLAN_TIMEOUT_SECONDS=180`。改为 `0` 并重启应用，即使用节点级期限；保留正值则启用整单应急上限。某节点超时会返回 `NODE_TIMEOUT` 与节点名，不能靠其他分支持续进度刷新它的期限。

</details>

<details>
<summary><strong>Chroma 或 Redis 不可用，是否完全不能规划？</strong></summary>

Redis 不可用时跳过缓存；Chroma 用户记忆返回 `skipped/degraded` 状态，仍可使用当前需求与 SQLite 画像。知识检索失败不能沿用旧证据。模型、候选或道路等核心依赖的失败会影响结果，具体状态由最终检查决定。

</details>

<details>
<summary><strong>这里的 checkpoint 能从任意节点崩溃恢复吗？</strong></summary>

当前应用快照用于已保存行程的修改，并未实现主 LangGraph 任意节点的持久化恢复。Trace 也位于进程内存。生产多实例部署前需要另行设计持久化执行状态、会话并发控制和断线恢复。

</details>

<details>
<summary><strong>MCP / A2A 是已部署的协议服务吗？</strong></summary>

目前是本地 `AgentCard`、工具 manifest 与任务 envelope 契约，用于演示能力发现、权限与消息结构。它们不是已部署的远程 MCP Server 或 A2A 网络服务。

</details>

## 文档与后续方向

- [部署与观测](docs/production-readiness.md)：健康检查、Token、节点指标和降级。
- [Supervisor 运行时](docs/supervisor-runtime-progress.md)：私有记忆、角色隔离与协作链路。
- [Reviewer 真实消融](docs/reviewer_ablation_v2.md)：对照方式、资源和失败归因。
- [RAG 评测协议](docs/rag_benchmark_v2.md)：语料、标注、冻结、指标公式与对照配置。
- [最终路线兴趣评测](docs/route_interest_evaluation.md)：只用实际选点验证知识是否有效。

下一阶段优先补齐道路证据缺失时的最终放行策略、候选问题的审核回检索路由、正式人工标注与冻结检索对照，以及持久化执行恢复。生产默认配置随证据调整，保留每版报告与失败案例。

## 项目来源与致谢

本项目基于 [FloatTrip](https://github.com/shouzhuoshouzhuo/FloatTrip) 开展课程与个人工程实践。原项目提供基础界面与旅行规划能力；本仓库重点扩展有界候选搜索、审核与返工、状态复用、道路核验、用户记忆、协作引擎及评测工具。具体改动以 [提交记录](https://github.com/yf-922/TravelMind-Agent/commits) 为准。

感谢 LangGraph、FastAPI、Chroma 及其他开源依赖。使用地图与模型服务时遵守各提供商条款；贡献代码前请移除真实密钥、用户数据和运行账本。

---

<div align="center">

**让每一步规划都有依据，让每一次修改都能继续。**

[查看源码](https://github.com/yf-922/TravelMind-Agent) · [反馈问题](https://github.com/yf-922/TravelMind-Agent/issues) · [查看 CI](https://github.com/yf-922/TravelMind-Agent/actions)

</div>
