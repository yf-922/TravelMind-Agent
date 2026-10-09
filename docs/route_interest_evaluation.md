# RAG 对最终路线选点的配对评测

## 当前状态（2026-10-08）

工具及离线验证已实现，正式实验尚未完成。不得将下面的资产数量或检索诊断称作最终路线提升。

- 30 条待人工确认请求，南京、上海、三亚各 10 条；开发/测试各 15 条。每条共享同城 6 个真实候选。
- 官方知识覆盖：南京 6/6、上海 6/6、三亚 6/6，共 18 条知识卡片。新增卡片均保留原文快照和内容哈希；上海豫园使用官方《游览须知》页面，仅记录官方购票渠道与游览安全事实，门票价格和开放时段仍需实时查询。新增来源包括南京文旅信息服务平台的科举博物馆、欢乐谷、中山陵页面，以及三亚文旅局的西岛和鹿回头页面。
- 18 个景点均有真实高德坐标，包括明确区分东馆/人民广场馆的别名核验。三轮补采后保存 90/90 条同城有向驾车路线，不填直线估算来凑齐。高德道路响应不等于包含船班、景区接驳或当天交通情况，海岛出行仍需实时核验。
- 使用固定 revision 的中文 BGE 做了 15 条开发请求的离线混合检索诊断，15 条均完成。**这只说明检索可运行，不是兴趣匹配正确，也不是路线对照结果。** 不调参、不查看未确认测试标注的表现。
- 5 条开发请求的真实规划预检已执行：账号用量接口能访问，但实际账号计费倍率尚未确认，因此在付费调用前停止。当前规划结果、Token、延迟及收益均未测得。
- 开发集检索诊断（15 条可回答请求）比较了混合检索的 `k=1/2/3/5`：`k=1` 候选 Recall 80%，`k=2` 为 93.3%，`k=3` 和 `k=5` 均为 100%；`k=3` 的候选 Precision 为 37.8%，`k=5` 降至 22.7%。因此路线小试暂定 `k=3`，这是探索性检索结果，不是最终路线质量指标。
- 生产主链路支持通过 `TRAVEL_INTEREST_KNOWLEDGE_ENABLED=1` 显式加载兴趣事实，默认关闭，检索模式仍由原有配置控制。最终路线收益尚未验证；实验 lookup 用于有/无 RAG 的严格配对对照。

## 对照流程与评分

两组使用完全相同的修复后生产联合 Planner、道路核验、风险门控、Reviewer、Time Check、餐馆结果整理、限时贴士和 Finalize 节点。入口固定在候选池核验之后，避免候选搜索差异干扰；不使用旧 Planner 子图。天气是相同的合成晴天场景，不伪称真实预报；驾车回放缺失明确报告。

最终路线完成耗时与可选贴士账单结算耗时分别保存。生产仍丢弃超时贴士的晚到结果；评测会等待该次调用结算，确保它的 Token 与费用计入所属试验。

唯一组间变化是 Planner 是否收到当前兴趣相关、城市和候选实体过滤后的知识。两组都使用原始请求优先、全程景点去重规则及相同结构化输出 Prompt。同一景点午餐前后重复时只保留第一段，下游继续检查空白天等问题，不把删重等同于合法路线。

只读取 `final_plan.days[].timeline` 中的 attraction 条目：

- 兴趣 Precision = 选中且可接受的不同景点数 / 最终不同景点数。
- 兴趣 Recall = 选中且可接受的不同景点数 / 标注可接受集合大小。集合允许多种可选路线；它不是“必须把集合全部排完”，Recall 只作描述，不作为唯一成功判据。
- 排除偏好违反率 = 最终选中明确排除景点的成功请求比例；失败率及总体通过率另报，避免用失败降低违反率。
- 硬约束由现有客观 grader 检查；同时报告核验是否完整。开放时间未知或地图缺失不能声称已证实满足现实约束。
- 返工次数为 Planner 执行轮数减一。各节点每次真实调用计量输入/输出 Token，包括内部重试、审核和贴士。

失败请求留在总体质量分母，另报成功请求质量、尝试可用率和执行覆盖率；没有执行的试验不冒充失败或成功。按意图组进行配对 bootstrap，展示选点改善、退化和无改善案例。相同实体的有效引用只表示引用关联正确，事实支持及未知请求拒答仍需人工核查，说明文字改进不算选点收益。

## 复现命令

```powershell
python scripts/review_route_interest.py validate
python scripts/validate_eval_assets.py
python -m pytest tests/test_route_interest.py tests/test_planning_knowledge.py -q
python scripts/compare_route_interest.py --retrieval-only --limit 15 --out data/route_interest/retrieval_dev.json
python scripts/compare_route_interest.py --out data/route_interest/preflight.json
```

采集或补充知识修改 `scripts/build_route_interest_assets.py` 的官方 URL 和可核验引文，再运行 `--collect`。已经人工确认的版本禁止覆盖，使用新的 `--out`。不得把同一个园区拆成多个虚假候选凑数。当前构建工具会保留已补齐的候选坐标；道路回放独立存储，变更候选后应重新采集和绑定。

```powershell
python scripts/capture_route_interest_maps.py --max-requests 40
python scripts/review_route_interest.py export --file data/route_interest/review.json
```

采集脚本限制 provider 操作次数；高德适配器单次操作最多两次尝试，实际 HTTP 请求可能多于操作计数。地图不消耗 LLM 预算。无法核验的响应保留缺失。

人工在导出的 JSON 中确认可接受/排除集合、必要来源，并填写 `annotator`、`reviewed_at`、`change_log`，将确认条目标记 `human_reviewed`。知识卡片也需要同样签署；机器不会代签。导入禁止改写 query、城市、候选池、分组、快照等，以防调参时改测试答案。

```powershell
python scripts/review_route_interest.py import --file data/route_interest/review.json
```

## 真实调用与总预算

先准备运行时 `data/route_interest/billing.json`：绑定模型、URL、API Key 的 SHA256 指纹（不是密钥本身）、24 小时内的 `verified_at` 和可核查 `evidence`；填写 `currency=CNY`、`input_per_million`、`output_per_million`、`account_multiplier`、`quota_units_per_CNY`。价格必须来自账户实际计费规则，不能猜倍率或把美元当人民币。

字段模板见 [route_interest_billing.example.json](route_interest_billing.example.json)。其中 `null` 表示尚未核验，模板不能直接执行付费实验。输入/输出价格单位均为人民币/百万 Token；账户倍率与价格分开记录，若账户展示的是最终折后价，应明确记录该口径并避免再次乘折扣。`quota_units_per_CNY` 是用量接口计量单位到人民币的换算，不是 Token 数量。`evidence` 应填写账户价格页或已核验截图的引用，不填密钥；`key_fingerprint` 由本机现有 Key 计算 SHA256。确认这些信息后再填写最近 24 小时内带时区的 `verified_at`。

```powershell
python scripts/compare_route_interest.py --execute --limit 5 --billing data/route_interest/billing.json --out data/route_interest/pilot_paid.json
```

共用 `data/route_interest/budget_ledger.json`，累计不超过 180 次初始规划、540 次实际模型调用、10 元。预算预留使用当前 Prompt 的 UTF-8 字节上界及 2048 输出 Token 上限，返回后按实际 usage 校正；每次调用前后检查网关 Key 用量。SDK 隐式重试关闭，节点重试逐次计量。计费未知、监控失效、未知 usage 或请求错误时停止。进程中断留下未核算调用时拒绝继续，先人工核对账单；不使用自动重跑掩盖已付费调用。账本使用排他锁阻止并发实验；异常退出的残留锁必须先核对账单再清理。

相同输入、知识、模型配置、道路回放和实现指纹的试验可以从账本复用，避免 5 条小试之后再付费重复第一轮；标注更新后对保存的最终状态重新评分。改变配置的小试仍计入总预算，不保证剩余预算足以完成全部组合；此时停止并报告未运行项目。

开发集小试链路及计费正常后扩大为 15 条 × 两组 × 3 次。知识覆盖、人工签署、完整道路回放达标后冻结：

脚本强制先完成同配置的 5 条开发请求、两组各一次的小试，才允许扩大请求数、重复次数或执行测试集。全量 pytest 本次结果为 390 项通过（见提交前验收）；它证明工程契约通过，不是模型质量指标。

```powershell
python scripts/compare_route_interest.py --write-freeze --freeze data/route_interest/freeze.json --limit 15 --repetitions 3
python scripts/compare_route_interest.py --execute --limit 15 --repetitions 3 --billing data/route_interest/billing.json --out data/route_interest/dev_paid.json
python scripts/compare_route_interest.py --split test --execute --limit 15 --repetitions 3 --freeze data/route_interest/freeze.json --billing data/route_interest/billing.json --out data/route_interest/test_paid.json
```

冻结绑定语料、全部标注、模型/检索/循环配置、回放与实现（含 Prompt）指纹。未人工确认或缺少完整证据时冻结会失败，禁止绕过。只有最终选点兴趣匹配改善且硬约束与可用率无明显退化，经过人工核查后，才考虑另行调整生产默认。

运行时 `data/` 不提交。开发诊断和停止报告可以提交，但其中不存在最终路线收益数字。
