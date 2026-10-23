# 四类局部修改：Checkpoint 与全量前置重跑的离线对照

2026-10-23，使用现有南京、上海、三亚各 10 份真实模型初稿状态，共 **30 份 checkpoint**。
每份初稿独立施加四类修改，共 **120 个修改探针**；每组每个探针重复本地计时三次，
因此每组 360 次计时，不是 360 次用户修改。原始初稿的天气、候选池来自固定 Provider fixture。

## 场景及结果

| 修改场景 | 探针 | 池内候选前提满足 | 局部定向补查 Mock 次数 | 全量 / 局部 Prompt 文本 Token | 总量缩减 |
|---|---:|---:|---:|---:|---:|
| 调整出发时间到 10:00 以后 | 30 | 30 | 0 | 73,678 / 51,731 | 29.79% |
| 将一个原景点替换成未选用的池内景点 | 30 | 30 | 0 | 76,308 / 52,257 | 31.52% |
| 下午改为池内室内景点避雨 | 30 | 20 | 0 | 77,428 / 52,481 | 32.22% |
| 新增点名的池外小众场馆 | 30 | 0，需补查 | 30 | 85,498 / 55,271 | 35.35% |
| 全部探针 | 120 | 80，无补查 | 30 | 312,912 / 211,740 | 32.33% |

三亚 10 份原候选池没有已标注的室内景点，因此对应避雨探针记录
`no_verified_indoor_candidate`，保留在总体统计，不计入缓存复用成功。
此基准没有运行 Planner 生成或拒答，不能把这些记录写成“最终正确拒绝”。

前三类中 **80 条候选前提满足的修改**，Prompt 文本 Token 为
**202,420 → 139,887，总量减少 30.89%**；每请求缩减比例算术均值 **30.90%**、P50 **30.73%**。
单独看有室内候选的 20 条避雨修改：**52,434 → 35,899，减少 31.53%**。

全量组每个探针分别执行一次天气和候选池回放 Mock，共 120 + 120 次；
局部组不执行这两个全量前置 Mock，只在 30 条池外点名请求中各执行一次定向补查。
局部补查执行 API 共用函数 `refresh_explicit_modification_candidates`，调用数取自 Mock 记录，
不根据期望结果填写。核验后两组候选池逐字段相同，源 checkpoint 保持不变。

## 统计范围

- Token 使用 `tiktoken / cl100k_base` 分词实际生产节点的 system/human 文本，
  公式为 `1 - sum(local_tokens)/sum(full_tokens)`。每请求比例的均值和 P50 另报，不能混淆。
- 全量组捕获 Intent、Query Rewrite、联合 Planner 的 Prompt；局部组仅捕获联合 Planner。
  全量组回放已解析的需求字段，不实际重新生成它们。两组都禁用知识检索注入。
- 补查 POI 明确命名为“离线Mock非遗手作博物馆”，坐标也是虚构 fixture；不代表真实景点，
  不作为知识库语料。生产补查只接纳同名且坐标有效的 Provider 返回，不接受模型编造。
- 只验证普通修改 API 的保守点名路由，不验证 Supervisor 的 LLM 修改分析，
  也不宣称模糊类别需求都能自动触发补查。
- Mock 请求数不是线上 HTTP 请求数，不能换算分页、重试或真实缓存命中率。
  未调用真实 LLM、地图或天气 API；未统计模型输出、计费 Token、端到端延迟或路线质量。
- 本地构造计时的全量 / 局部 P50 为 **0.6172 / 0.2983 ms**，仅含序列化、状态恢复、Prompt 构造；
  所有补查 Mock 在重复计时之前执行，不包含在该耗时中。

## 复现与证据

```powershell
python scripts/offline_engineering_benchmark.py --runs 10 --checkpoint-samples 30 --checkpoint-repetitions 3 --json-out data/eval/checkpoint_modifications_v2.json --markdown-out data/eval/checkpoint_modifications_v2.md --prompt-out data/eval/checkpoint_modifications_v2_prompts.json
python scripts/validate_eval_assets.py
pytest -q tests/test_checkpoint_modifications.py tests/test_offline_engineering_benchmark.py tests/test_evidence_benchmarks.py tests/test_manual_edit_api.py
pytest -q
```

逐条指令、候选前提、源状态哈希、两组候选池哈希、Mock 计数、补查结果、分节点 Token 与计时
保存在 `data/eval/checkpoint_modifications_v2.json`；完整 Prompt 保存于对应 `_prompts.json`。
运行数据受 Git 忽略，复现需要本地原账本 `data/eval/reviewer_ablation_real_v2.json`；
仓库测试使用独立的离线小账本，无需私人账本或 API Key。

本次记录绑定：

- 账本 SHA-256：`1fd90eeca8e10abfcabad13ce9702809ef4619eeafa3736359a91d055f0c9045`
- Prompt SHA-256：`c4f273c9796d22d752daa6b479516110ebf4f5971f083d71478216a9b08e4cbc`
- 执行源码指纹：`72e5d00a58d0eddfeac98c9e63e7312c6b9e8e81ca31824b9b4cdfc75ab527e7`

复现时本地耗时会波动；Token 结果也会随生产 Prompt、账本或 tokenizer 版本变化。

验证记录：评测资产校验通过；专项测试 **61 passed**；全量 pytest **537 passed，26.19 秒**。
