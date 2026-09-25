# TravelMind-Agent 面试项目说明

这份说明用于把仓库事实转换成简历和面试中的可验证表达。数字只引用本地测试或部署输出，不把离线 Fixture 成绩包装成真实线上效果。

## 一句话介绍

基于 FastAPI + LangGraph 的多智能体旅行规划服务：把用户意图识别、POI 检索、行程规划、Reviewer 反馈、开放时间核查、餐饮推荐和景点贴士串成可观测的 SSE 工作流，并用 SQLite/Redis/Chroma 保存用户画像、历史行程和语义记忆。

## 可直接放进简历的项目描述

### AI 应用开发方向

- 设计并实现 LangGraph 多智能体编排链路，拆分 Intent、Query Rewrite、POI Search、Planner、Reviewer、Time Check、Meal Recommendation 等节点；通过结构化 Pydantic 输出、候选池封闭约束和 Reviewer 反馈循环降低景点幻觉与不可执行路线。
- 构建 SSE 流式进度协议，前端可实时看到节点开始、结构化阶段摘要和最终结果；为每次规划生成 `run_id`，记录节点尝试次数、耗时、失败码、慢节点、并行重叠、降级服务和 request-context LLM 用量，并提供 owner-only Trace API 与 Prometheus 指标。
- 建立离线可复现评测：当前完整 Python 测试套件 `178 passed, 1 warning`；RAG 30 条检索集与记忆隔离集均已产出离线报告。30 条在线场景的 POI 已从高德 Place Text API 冻结为 fixture，并通过来源、时间戳、坐标、评分、去重和室内标签校验；已完成 39 条结构消融案例、真实三模式回放和故障恢复，但不把单次结果包装成稳定性成绩。
- 为在线评测增加 Dry Run 和双重预算门禁：显式区分图中的逻辑 LLM 调用与重试放大的供应商请求尝试；完整 Planner/Reviewer/Time Check 在 30 条、每条 5 次且启用 Judge 时，上限为 2,250 次逻辑调用 / 6,750 次供应商尝试，默认拒绝直接执行。
- 将模型路由接入统一 LLM 工厂：Intent、Query Rewrite、画像/记忆抽取等有界任务可选同供应商小模型，Planner、Reviewer、Judge 保留主模型；默认关闭且显式模型覆盖不丢失。当前只完成离线契约验证，不虚构多模型线上 A/B 收益。
- 补充平台化最小协议原型：Agent Card、MCP Tool Manifest、A2A Task Envelope 统一能力发现、工具权限和跨 Agent 结构化消息，但不虚构远程协议部署。
- 将 RAG 改为可解释的双路召回：Chroma 语义候选与本地关键词候选独立取 Top-N，再用 RRF 融合、去重并保留 `vector_rank`、`keyword_rank`、`retrieval_channels` 和引用字段；Chroma 不可用时显式降级为 keyword-only。新增最低词法相关度和弱向量最近邻拒答，避免知识库外问题被强行匹配；30 条离线集含 6 条 no-answer，不能把检索成绩包装成生成答案质量。
- 将“少走路”和“雨天优先室内”从自然语言偏好升级为结构化约束；Planner 出稿后并发调用高德步行/驾车路线 API 核验实际道路距离，再把结果交给 Reviewer，接口失败明确标记未核验；修改行程也重新经过道路距离、Reviewer 与 Time Check，手动优化则基于有向驾车距离矩阵，不使用直线距离伪装道路收益。
- 将无依赖的 Query Rewrite 与天气查询改为 LangGraph fan-out 并行分支，并以多前驱屏障汇合后进入 POI Search；路线中的多段道路查询使用有界线程池并发，专项测试验证两个图分支确实存在时间重叠。
- 增加 Planner / Reviewer / Time Check 消融评测：将原先 3 条示例扩展为 39 条分层案例（正常控制、候选池越界、重复景点、开放时间冲突及不可修复草案），总体通过率从 23% 提升到 67% 再到 90%，平均调用节点从 1.0 增加到 2.0/3.0；该结果只用于解释硬约束编排收益，不冒充真实 LLM 质量。
- 增加真实 LLM 在线消融入口：在完全相同的 Fixture 和模型配置下切换 Planner Only、Planner+Reviewer、Planner+Reviewer+Time Check，逐模式统计通过率、逐案例 `pass@k/pass^k`、Judge、误放行、节点调用、Token 和端到端 P50/P95；首轮三案例结果已落盘，但每例仅一次，不宣称统计稳定性。
- 完成首轮真实单/多 Agent 对照：3 个自然案例各跑 1 次时三种模式硬约束均为 3/3，Judge 均值为 4.47/4.33/4.27，完整多 Agent 的估算 Token 与平均延迟约为单 Planner 的 2.6 倍/2.0 倍，因此不能宣称固定多 Agent 更优。另在相同真实 Planner 草案上注入重复景点和时间故障，无审核恢复 0/2，真实 Reviewer 回环和完整链路均恢复 2/2；结论是审核适合风险触发与失败恢复，而非所有请求无条件执行。
- 将上述结论落到主流程：道路距离核验后执行零 LLM 的确定性风险门控，重复/越界、结构与习惯冲突、雨天露天、步行约束、超长道路段和用户修改才升级 Reviewer，已知开放时间冲突才升级 Time Check；开放时间未知时不让模型猜测，标记 `partial` 并提示用户核实。保存真实输出回放中自然路线 3/3 跳过、注入故障 2/2 升级，按三案例实测均值投影可减少约 61.8% Token、50.7% 延迟；该节省是回放投影，不冒充线上 SLA。
- 将 POI Agent 的真实调用统一收口到工具权限注册表，输出不含原始查询的审计轨迹；离线轨迹评测覆盖正常链路、Reviewer 打回、工具瞬时失败后重试、持续失败后结构化终止，当前 4/4 场景各通过 8/8 项合约。该结果只证明编排、权限、参数和失败处理，不代表线上规划质量。

### 后端开发方向

- 使用 FastAPI + SQLite + Redis 实现登录、行程历史、手动编辑、路线优化、缓存和健康检查；Redis 采用 Cache-Aside，外部依赖不可用时保留可用的降级路径。
- 增加请求级 `X-Request-ID`、HTTP 平均/P95/最大延迟、Agent 节点级耗时、LLM token/调用/延迟/成本指标、单次规划超时和错误脱敏；供应商 usage 可用时记录真实 token，否则明确记录估算值与 usage 精确率，Trace 不保存 Prompt、模型思考和模型输出。
- 提供 Docker Compose 一键启动 API + Redis，已实际构建镜像并验证 `/api/health` 返回 SQLite/Redis `ok`；GitHub Actions 执行编译检查和完整离线测试集。

## 可验证证据

| 证据 | 命令或接口 | 当前结果 |
|---|---|---|
| Python 回归 | `python -m pytest -q` | 170 passed, 1 warning |
| Python 编译 | `python -m compileall -q app tests scripts` | 通过 |
| 编排消融 | `python -m tests.eval.ablation --json evaluation/ablation_report.json --out evaluation/ablation_report.md` | 39 条分层案例：planner_only 23% → planner+reviewer 67% → +time_check 90% |
| 容器构建 | `docker compose up --build -d` | 镜像构建成功 |
| 依赖健康 | `GET /api/health` | database=ok, redis=ok |
| HTTP/Agent 指标 | `GET /api/metrics` | Prometheus 文本格式 |
| 执行追踪 | `GET /api/agent-runs/{run_id}` | owner-only、慢节点、并行重叠、降级状态和单次 LLM 用量 |
| 评测资产校验 | `python scripts/validate_eval_assets.py --require-fixtures` | 30/30 fixture 就绪，共 523 条场景内 POI 记录；来源为高德 Place Text API v3，天气为人工冻结场景 |
| RAG 模式/拒答 | `python scripts/evaluate_rag.py`；同一 30 条集合的 hybrid 对照 | 24 条正例 + 6 条 no-answer：keyword Hit@1/Recall@3/拒答准确率均 100%；hybrid 为 95.8%/100%/100%；只证明三文档离线检索 |
| 在线预算预检 | `python -m tests.eval.run_eval --dry-run` | 只计算逻辑调用、供应商尝试上限和 Fixture 就绪状态，不访问网络 |
| 在线单/多 Agent 消融预算 | `python -m tests.eval.run_online_ablation --max-cases 10 --k 3 --dry-run` | 扩展到 10 案例 × 3 次前的最坏供应商尝试上限为 2,430；当前只执行了 3 案例 × 1 次 |
| 首轮真实单/多 Agent 对照 | `evaluation/real_single_vs_multi_ablation.md`；`evaluation/real_fault_recovery_ablation.md` | 自然请求未观察到多 Agent 质量收益；同稿故障恢复为 0/2 → 2/2，但仅 2 个注入场景、每例 1 次 |
| 风险门控策略回放 | `python scripts/evaluate_risk_gate.py` | 保存真实输出上自然路线跳过 3/3、注入故障升级 2/2；Token/延迟节省 61.8%/50.7% 为投影 |
| Agent/工具轨迹 | `python -m tests.eval.run_trajectory_eval` | 4 类故障场景、每类 8 项契约检查；校验调度顺序、权限、脱敏参数、重复调用、重试边界、终止状态和候选池传递 |
| 模型路由/协议契约 | `python -m pytest tests/test_model_router.py tests/test_agent_protocols.py -q` | 9 passed；路由已接主流程，协议仍是本地原型，不等同远程 MCP/A2A 服务 |
| 在线链路基准 | `python scripts/benchmark_pipeline.py --runs 3 --allow-external-calls` | 输出配置指纹、成功率、P50/P95/最大延迟、Token/成本和节点耗时增量；新数据待明确预算后实测 |

在线冒烟使用本地 `.env.local` 中配置的 Grok 兼容接口；脚本只输出事件数量、状态、错误码和耗时，不输出 API Key、Prompt 或行程正文。

## 面试演示顺序（约 3 分钟）

1. `docker compose up --build -d`，打开首页或调用 `/api/health`。
2. 展示规划 SSE 的 `run`、`stage`、`stage_summary`、`result` 事件，以及响应头里的 `X-Agent-Run-ID`。
3. 查询 `/api/agent-runs/{run_id}`，解释 Planner/Reviewer 循环和各节点耗时。
4. 临时停止或不配置 LLM Key，展示错误码归一化、超时/失败状态和 `/api/metrics` 的计数变化。
5. 打开离线 RAG/记忆/并发报告、真实三模式消融、故障恢复报告和 `eval_asset_report.json`，说明当前真实结果每例仅一次，30 条完整稳定性评测尚未执行。

## 需要本人补充、不能代写的内容

- 你实际负责的模块和投入时间；
- 真实线上/本地压测数据（并发数、P95、错误率）；
- 如果使用了真实 API Key，按服务商账单记录填写 Token 和成本；
- Git 提交记录、Issue/PR 链接和演示视频地址。

## 当前诚实边界

离线 Fixture 的通过率只证明编排和结构契约，不代表真实 LLM 质量。39 条消融案例中仍保留不可修复草案，因此完整链路为 90% 而非 100%；异常输入集用于验证安全边界，不应写成业务成功率。30 场景的 POI 来自一次高德抓取，但天气是为可复现评测人工冻结、室内属性是名称关键词启发式标注；真实 LLM/Judge 目前只有 3 个单案例冒烟，不能代表 30 条集合或 5 次重复稳定性。简历中应分别写“评测数据已构建”“离线契约结果”和“在线 smoke 结果”，不要合并成单一线上成功率。
