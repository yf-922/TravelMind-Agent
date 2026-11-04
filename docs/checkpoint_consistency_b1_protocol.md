# B-1 生成重放预注册

B-0 确认 120 个探针的全量与增量路径 Prompt 结构不同：全量包含 Intent/Query Rewrite/Planner 前置节点，增量只调用 Planner，并携带已保存 checkpoint 上下文。两条路径的候选池哈希逐探针一致。B-1 因此不能把两条 Prompt 当作同一输入的重复采样；主分析比较 `A(F,I)`，并在存在随机解码时加入同一输入的 `F(seed k)`/`F(seed k')` 噪声基线。

## 冻结臂

正式启动前必须在 `b1_binding.json` 写入基座/Checkpoint 路径哈希、vLLM 版本、temperature、top_p、max_tokens、seed、batch/concurrency 和 prefix-cache 设置。当前 v2 账本没有保存这些字段，故本文件只完成预注册，不虚构具体模型或解码配置。

### B-1-launch 考古结论（2026-10-30）

只读扫描脚本为 `scripts/audit_b1_artifacts.py`，结果见
`data/eval/checkpoint_prompt_diff_b0/b1_manifest.json`。当前仓库能证明一套**新的**主评测协议：Qwen2.5-7B-Instruct 系列、完整 BF16、`temperature=0.3`、`top_p=0.95`、`max_tokens=2048`、共同 seeds `1001/1002/1003`、并发 1、关闭 Prefix Cache，返工 seed 偏移 10000。这些配置来自当前受控评测脚本，不能回填为 2026-10-23 B-0 运行时配置。

B-0 历史账本仍缺 Planner 实际 checkpoint/model、解码参数、vLLM 版本和启动命令，因此 `launch_allowed=false`，B-1 生成不得启动。A 评测也没有本地历史 manifest；仅有远端 adapter binding，不能替代运行 manifest。

### B-2 结果边界（2026-10-30）

B-1 已按新声明运行 360/360 条，但请求使用通用 `json_object`，没有强制 `TravelRoute` Schema。三臂输出均可被 JSON 解析，却缺少 `days/spots`；因此 F@1001 与 I@1001 的 120 条只能记为“字节相同但路线 Schema 不可用”，不能记为语义等价。F@1001 与 F@1002 的 120 条也无法进行路线语义比较。规则核验和失配 ×3 仲裁在零模型调用的 B-2 中未执行，并已在 `b2_analysis.json` 标记原因。

### Invalid run: B-1 output-consistency replay (2026-10-30)

该批次标记为 **invalid-run**。零推理审计确认：重放客户端把 LangChain 的 `role: human` 原样发送到 OpenAI 兼容 vLLM；Qwen Chat Template 只识别 `user`，因此需求、checkpoint 上下文和修改请求被静默丢弃。服务端每条请求只收到固定 253-token system 壳，而 B-0 账本中的 Planner 输入随探针变化，为 1,300–1,800 tokens。单 seed 内所有输出逐字节相同，换 seed 后整体变成另一份固定输出，故本批次不产生路线级结论。

B-1b 延后修复为 `human → user`。启动门禁冻结为：输入完整性 dry-run 中服务端 `prompt_tokens` 需与完整 Prompt tokenizer 计数一致；3 条探针必须通过 `TravelRoute` Schema；跨探针输出哈希碰撞立即中止，并记录请求序号、时间戳和 response ID。

- 确定性臂：全量 120 + 增量 120，固定 seed 和 greedy/确定性解码；若同一 Prompt 多次结果不一致，逐条最多重放 3 次，仍不稳定则标记引擎非确定性。
- 采样臂：只有 B-0 发现 Prompt 存在会影响生成的实质差异时启用：`F(seed k)`、`I(seed k)`、`F(seed k')`。主结论比较两条路径与全量自身噪声基线，避免把采样波动归因于增量路径。
- 所有原始响应、usage、finish reason、错误和请求 Prompt 与 Prompt 对同级落盘；不只保存聚合指标。

## B-2 指标

逐对报告字节级、归一化 JSON 等价和行程语义等价；行程语义定义为逐日景点名称、开始时间、结束时间顺序一致。两臂输出再用同一确定性规则评分，报告全量通过/增量通过、双通过、全量单独通过、增量单独通过。只有“增量规则通过率不低于全量”才支持安全性结论；文本或语义一致率不能替代规则通过率。
