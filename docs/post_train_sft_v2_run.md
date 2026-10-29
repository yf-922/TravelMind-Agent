## 2026-10-24 17:43：补充数据审计完成，正式SFT-v2已启动

96条成都/苏州补充样本已合成完成：97次教师调用，95条首轮通过、1条通勤返工通过，实际433,491 API Token；全局已启动519次付费调用（含历史中断），本轮不再追加教师请求。

复核覆盖全部96条新增数据及原156条训练数据：生产messages精确捕获一致、Schema、候选归属、开放/最晚入场、已保存真实道路、日结构、跨天重复、天气与rubric、输入去重、验证/城市留出隔离及长度。新增解释检查区分精确时长、至少/超过/至多、时长范围与景点之间的转场时间，避免把“预留40分钟交通”错认成游览40分钟。隔离newcity-苏州-019（正文每个90、实际105）及newcity-苏州-043（正文每个75、实际90），不修改教师答案伪装一致。原156条训练保持，新增94条，共250训练/30验证；验证账本SHA256与v1完全一致。新样本分布：普通16、多日23、高温24、时间窗口23、暴雨室内8。训练最长2488 Qwen tokens，无截断。两个独立测试城市仍未用于训练。

额外逐条阅读29条新样本的reasoning、notes与时间表，覆盖两城市全部3日案例、自然修正案例及普通/高温/窗口/暴雨分层样本。解释语义仍非穷尽证明，也不伪称人工专家认证。完整审计/隔离原因与阅读清单保存在data/post_train/release_v2/audit.json及agent_review.json。

250条正式训练已在/home/jiale/TravelMind-posttrain启动，GPU0，从原始Qwen基座重训，3epochs、梯度累积16、48优化步，按原30条验证集最低loss选检查点。新版输出独立saves/TravelMind-Planner-SFT-v2及models/TravelMind-Planner-7B-SFT-v2，旧SFT-v1不覆盖。上传6个数据/审计文件均通过SHA256核验。启动器首个尝试因状态字段重复报错，未开始训练；已修复并新增回归测试，失败日志与启动标记保留为attempt1，成功重启后日志确认250样本/48步。

后台验证任务已接续：等待SFT-v2训练及CPU合并完成，临时加载新版，使用相同30条开发输入、temperature0.3、seed42、JSON Schema和2048输出预算评测；Base与SFT-v1沿用原始保存记录，比较请求中除模型名外的参数完全一致，所有失败均入分母。结束后恢复Base服务。结果位于data/post_train/sft_validation_v2，报告docs/post_train_sft_v2_validation_report.md。不能把训练规则通过率当最终模型效果。

DPO不自动启动。下一步先看相对SFT-v1的配对改善/退化、重复地点/天气/时间窗/通勤、输出截断和解释一致性，再确定DPO是否值得做。丽江/景德镇独立测试留至协议与模型冻结后，最终四组Base/SFT/SFT+DPO/API采用统一输入和预算分别报告Planner-Only与Rules+Rework。

训练状态：服务器logs/sft-v2-status.json；训练日志logs/sft-v2-training.log；显存/温度logs/sft-v2-gpu.jsonl。开发验证状态：本机data/post_train/sft_validation_v2/job_status.json。初估训练+合并10–20分钟、后续开发验证5–10分钟，以实际步速及热限速为准。
