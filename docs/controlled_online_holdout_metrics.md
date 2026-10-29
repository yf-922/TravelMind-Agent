# 受控线上闭环评测：指标和分析模板

本协议覆盖冻结的 50 场景，比较 Base、v2-SFT、A-SFT、B-SFT、v2-DPO、A-DPO、B-DPO 七组。每组 50 场景 × 3 个共同采样 seed；每个 case/seed 只执行一次 Rules+Rework 闭环，并从同一轨迹投影 Round 0 作为 Planner-Only，因此输出 12 个汇总行，但只有 1,050 条闭环轨迹（另有每组 5 条丢弃的 Planner warm-up）。

## 固定控制条件

- 权重统一完整 BF16：SFT 和 DPO LoRA 依训练父子顺序合并后再加载；Base 与 v2-SFT 已是完整权重。禁止在本表混入运行时挂载 LoRA 的延迟。
- 每组顺序加载单卡 vLLM，随机化组顺序；单 GPU 无法同时驻留所有 7B BF16 权重，因此不声称跨模型逐题同时间交替。模型服务启动使用 `--no-enable-prefix-caching`，每组结束杀掉本次自有进程，等待至少 4 秒并轮询 `nvidia-smi` 至显存回落后才启动下一组。
- Planner 采样固定 `temperature=0.3, top_p=0.95, max_tokens=2048`、并发 1。每个基础场景重复使用共同 seed `1001/1002/1003`；同一 seed 对应所有权重，重规划时用 `seed + 10000 × Planner轮次序号`，保证跨模型对齐。
- 每组正式请求前，以 5 个场景执行 Planner-Only 预热；预热结果丢弃并在 `warmup.json` 报告 usage，不纳入 150 条指标。
- 权重按模型滚动合并：一次只创建当前 BF16 权重；A/B-DPO 重建训练时记录的 SFT 父模型后叠加 DPO adapter。合并前按源权重大小预测单输出/父子双输出峰值，生成权重总量上限 40 GB 且需额外保留 8 GB 空闲；合并后复核实际目录。模型评测完、vLLM 关闭且 GPU 显存回落后，立即删除本次创建的 BF16 输出和临时父模型。原始模型和轻量 adapter 保留。
- 无解题不触发额外的评分器反馈。每个 case/seed 在一次闭环轨迹后生成 Planner-Only 投影，因此不会额外执行一遍首轮 Planner。
- 提示词、Pydantic Schema、风险门控、道路节点、Reviewer、时间核查、归一化、最终门控使用项目生产代码。唯一图行为差异是全局最多再调用 Planner 一次；Reviewer 与 time_check 不改、不限流、不替代。
- Reviewer/time_check 固定现有线上 OpenAI 兼容/Responses API 配置。其成本包含在 Rules+Rework 的总 Token。API 服务重试的真实账单可能包括未返回 usage 的失败请求。

## Quality 指标

|指标|分子/分母|
|---|---|
|Feasible Delivery Rate|45 个可行题中，Planner 输出成功解析并通过生产 TravelRoute Schema 的比例|
|Feasible All-Rule Pass|45 个可行题中，交付且独立确定性 rubric 无规则错误的比例；未交付计失败|
|Overall Delivery / All-Rule|另以全部 50 题为分母提供总览；Overall All-Rule 不替代 45 题可行集的主质量指标|
|In-Pool Faithfulness|可行题中池内景点实体数 / 全部被规划景点实体数，按实体 micro 平均；无景点实体时为未定义|
|Opening Hours Pass|可行、成功交付题中所有景点时间满足冻结开放窗口及最晚入场 / 这些题数|
|Timeline & Transit Pass|可行、成功交付题中时间不重叠且所有相邻行程段有冻结道路证据并留足通勤时间 / 这些题数|
|Non-Duplicate Pass|可行、成功交付题中全行程无重复景点 / 这些题数|

Feasible Set 固定为 45 题：可行 Delivery、All-Rule、路线子规则及可行任务延迟分布使用这组题，子规则未交付计失败。Infeasible Set 固定为 5 题，单独报告每模型 15 次运行的不可行识别率、拒答合规率、路线幻觉率和 E2E 分位数。无解题通过要求：TravelRoute 合法；保留完整天数；所有 `spots` 与 `meal_slots` 为空；说明中出现不可行表述并命中该题证明对应的冲突原因。正确拒答在 `correct_task_rate` 中算任务成功，但不算路线 All-Rule 通过。`online_approval_rate` 和 `correct_task_rate` 独立呈现闭环结果。Reviewer 对软偏好作否决不会伪装成交通或开放时间规则失败。

## Rework 与稳定性

- 实际 Rework Trigger Rate：生产图真实执行了额外 Planner / Rules+Rework 任务数。规则失败触发代理率：配对 Planner-Only 中 All-Rule 未通过的比例；门控可能因独立 Review 而返工或拒绝，因此二者不要求相等。
- Avg. Rework Turns：Rules+Rework 中 Planner 调用数减一的均值，上限 1。
- Rework Recovery Rate：按同场景、同 seed 配对；首轮 All-Rule 失败中闭环 All-Rule 通过的比例。另统计首轮已通过而返工后失败，避免只看挽回、不看回归。
- Pass@1 mean ± std：每个 seed 对 45 个可行题的规则成功率独立计算，报告三个采样 seed 的均值与标准差（百分点）。
- All-3-Pass：45 个可行场景中三次都通过的比例；附 `0/3,1/3,2/3,3/3` 计数。无解场景另报三 seed 识别稳定率。

## Token 与延迟

- 报告所有角色总 Prompt/Completion/Total Token；另报 Planner 专属平均 Completion Token 和 Planner 原始输出字符数，以比较 SFT/DPO 长度。
- 所有模型调用的 E2E 任务延迟 P50/P95 包含真实 API Reviewer/time_check，但 Feasible 与 Infeasible 分组分别计算；预热不计入。总 Token 包括重规划级联消耗。
- 瞬时网络错误最多退避重试两次，记录异常类型、HTTP 状态、次数和等待秒数；重试等待时间从 E2E 与 TPOT 活跃时间统计中扣除。失败请求未返回的 Token usage 无法推算。
- Planner 使用 `stream=True`，取得首个非空文本 chunk 时间 TTFT；TPOT 为 `(Planner流总耗时 - TTFT)/(输出completion token - 1)`。它含请求往返与服务排队，是服务端到端 normalized TPOT，不是独占 GPU kernel 速度。
- Prompt/Completion usage 缺失不估算。以供应商账单核对 API 金额，不从 Token 反推未配置的价格。

## 结论写作模板

> 在 50 个固定场景、同一生产 Planner 提示词与 Schema、生产规则门控及 Reviewer/time_check、相同的一次额外 Planner 重试上限下，DPO 相对对应 SFT 的 Planner-Only All-Rule Pass 从 **X%** 变为 **Y%**（Pass@1 mean±std：**…**，All-3-Pass：**…**）。提升主要来自 **Timeline & Transit / Opening Hours / In-Pool** 中的 **…**；**Completion Token** 从 **A** 变为 **B**（**C%**），没有/存在可观察输出长度变化。Rules+Rework 平均额外 Planner 轮次从 **R1** 变为 **R2**，总角色 Token 从 **T1** 变为 **T2**（**Z%** 差异），配对挽回率 **Q%**，同时报告返工回归 **…**。Planner TTFT/TPOT 的中位数为 **…**，E2E P50/P95 为 **…**。这些结果支持/不支持 DPO 在该冻结集上的质量与成本收益；它们不是跨城市总体显著性结论，因为只有 10 城市且权重按单卡顺序评测，模型间延迟仍可能受时间漂移影响。

不得把 `rules_pass` 同时称作常识分或软约束分。自动类别偏好和解释代理只作为单独诊断，软约束没有通用、可靠判分器时不纳入硬规则指标。
