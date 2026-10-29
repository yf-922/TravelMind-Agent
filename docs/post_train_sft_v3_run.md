# SFT-v3受控修正实验

2026-10-24启动。用户授权按审阅、版本冻结、受控SFT、重复开发评测推进；不自动DPO。

## 数据与审阅

32条补数逐条读取reasoning、notes、选点解释及最终时间表，结合固定开放、道路及室内范围证据。排除5条：成都explanation-1合计150而90+75=165；苏州sparse-0/sparse-1/multi_day-0把“10:00后”排成10:00；成都multi_day-0“60分钟以上”实际60存在严格语义歧义。旧synth-0127返工语境泄漏排除。源ledger不改，决策记录到release_v3/agent_review.json。

249条历史保留目标重新通过新规则审计，重新捕获当前生产输入，保存original_messages和prompt_origin=audited_contract_migration_of_historical_teacher_target_not_new_teacher_call。没有额外教师调用，不把迁移消息冒称原始教师请求。27条补数保留实际调用边界messages。训练276条，5城市；最大3021tokens，小于3584。

历史249条做了全量确定性重放，并非逐条重新完成全部语义认证；补数主要覆盖稀疏候选、去重、室内三值及高温边界，对复杂跨日道路优化支持有限。独立测试丽江/景德镇没有参与选择或训练。

验证loss数据30条与v2完全相同，保留旧输入以维持连续性；其loss不是新协议部署指标。实际生成评测使用新生产协议输入。

## 训练协议

从原始Qwen2.5-7B-Instruct重新训练，不继续v2权重。LoRA16/32/all，lr1e-4，bf16，SDPA，gradient checkpointing，batch1×accum16，cutoff3584，seed42，max_steps48。总优化步数与v2相同，但训练集276而非250，等价epoch暴露不同；不能宣称完全单变量实验。max_steps覆盖epochs，实际以48步为准。保持按epoch保存与旧验证loss选checkpoint策略。

GPU0，路径均/home/jiale；离线依赖与本地权重，不从服务器下载模型。v1/v2模型和数据保留。saves/TravelMind-Planner-SFT-v3及models/TravelMind-Planner-7B-SFT-v3为全新目录，hash锁拒绝覆盖。

## 自动后续

训练完成合并CPU权重，先恢复Base服务；本机等待脚本随后加载v3，运行固定30开发状态×种子42/43/44=90次，temperature0.3、max_tokens2048，与已录v1/v2请求除model名外逐字核对。评测后恢复Base服务，汇总docs/post_train_sft_v3_result_review.md。

优先判断规则+解释联合成功、三次全过的稳健案例数、多日重复率、高温/室内错误、开放与通勤错误；JSON解析仅最低门槛。重复生成不能当90个独立问题，新协议也不能直接与旧协议18/30和17/30拼接。不得仅据训练loss下降自动进入DPO。

服务器日志：logs/sft-v3-status.json、sft-v3-training.log、sft-v3-merge.log、sft-v3-gpu.jsonl；本机评测data/post_train/sft_contract_eval_v3。训练预计15–25分钟、生成评测8–12分钟，实际以日志为准。

验证：591 pytest通过，3条既有依赖警告。
