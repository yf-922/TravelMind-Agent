# TravelMind-Agent 面试项目说明

这份说明用于把仓库事实转换成简历和面试中的可验证表达。数字只引用本地测试或部署输出，不把离线 Fixture 成绩包装成真实线上效果。

## 一句话介绍

基于 LangGraph 的 LLM 多 Agent 旅行规划服务：把用户意图识别、POI 检索、行程规划、Reviewer 反馈和开放时间核查串成可观测的 SSE 工作流，并用 SQLite/Chroma 保存用户画像、历史行程和语义记忆。

## 可直接放进简历的项目描述

### LLM 应用开发方向

**项目时间：** `请填写实际时间` ｜ **个人仓库：** [TravelMind-Agent](https://github.com/yf-922/TravelMind-Agent) ｜ **原始项目：** [FloatTrip](https://github.com/shouzhuoshouzhuo/FloatTrip)

- **多 Agent 编排与局部重规划：** 原 FloatTrip 流程在用户改路线时从头执行，重复检索且上下文容易相互污染；我基于 LangGraph 对比并选择显式 `TravelPlanState` + checkpoint，把 Planner、检索 Worker 和 Reviewer 拆成独立节点。修改意见到达后恢复候选池，仅重跑受影响节点；候选不足时才定向查询，避免用固定状态机覆盖所有意图。
- **输出质量与坏案例治理：** 针对“重复景点、候选池越界、开放时间冲突、步行距离超限”等失败样例，采用“确定性风险门控先筛查，低风险跳过 LLM Reviewer，高风险进入审核修复”的分层方案；用 39 条固定案例做消融，Planner-only、+Reviewer、+Time Check 分别通过 **9/39、26/39、35/39**，保留不可修复样例而非报成 100%。
- **记忆、RAG 与交互取舍：** 用户偏好既要可精确更新又要支持语义召回，因此选择 SQLite 保存结构化事实、Chroma 保存按 `user_id` 隔离的语义记忆，而不是只用向量库；按段落切块并做向量/关键词混合召回，Pydantic 校验 Agent 消息，FastAPI + SSE 流式返回节点进度。
- **工具调用与性能权衡：** 原流程用直线距离判断可达性，容易放过真实步行不可达路线；改为高德步行/驾车路线核验，并行执行天气、意图改写和路线查询。6 条保存的真实 API 响应回放中，自适应审核平均延迟较全量审核降低 **21.6%**；3 条自然请求的 Token 为估算值，减少 **61.8%**。两项均为小样本机制验证，不代表线上 SLA 或实际账单。
- **可靠性与工程落地：** 从地图不可用、LLM 超时、开放时间核查失败等故障轨迹反推降级状态、重试边界和错误码；补充 4 类故障轨迹、32 项流程契约测试，提供 Docker Compose、Swagger、评测脚本和真实响应回放，GitHub Actions 执行 `pytest`，当前 **178 条测试通过**。

### 后端开发方向

- 使用 FastAPI + SQLite + Redis 实现登录、行程历史、手动编辑、路线优化、缓存和健康检查；Redis 采用 Cache-Aside，外部依赖不可用时保留可用的降级路径。
- 增加请求级 `X-Request-ID`、HTTP 平均/P95/最大延迟、Agent 节点级耗时、LLM token/调用/延迟/成本指标、单次规划超时和错误脱敏；供应商 usage 可用时记录真实 token，否则明确记录估算值与 usage 精确率，Trace 不保存 Prompt、模型思考和模型输出。
- 提供 Docker Compose 一键启动 API + Redis，已实际构建镜像并验证 `/api/health` 返回 SQLite/Redis `ok`；GitHub Actions 执行编译检查和完整离线测试集。

## 可验证证据

| 证据 | 命令或接口 | 当前结果 |
|---|---|---|
| Python 回归 | `python -m pytest -q` | 178 passed, 1 warning |
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
