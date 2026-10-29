# 扩池教师答案合成进度

更新时间：2026-10-25T00:26:21.651010+08:00。

当前生产 Joint Planner 已明确纯 JSON、固定字段、仅输入快照、禁止工具调用和简洁解释；通过真实生产函数捕获消息。旧答案和评测答案没有提供给教师。未启动训练或 DPO。

|项目|数量|
|---|---:|
|母本输入|250|
|可生成输入|235|
|输入证据隔离|15|
|已完成记录|14|
|新审计下接受|12|
|剩余输入|221|
|当前版本实际调用|7|
|整个扩池实验实际调用|38|
|全局累计实际调用|654|
|扩池实验接口报告总 Token|469115|

缺少用量响应的失败调用：2 次，已计入调用预算；报告Token是已返回部分，不能据此断言失败调用零成本。

输入隔离原因：{"no_scoped_verified_indoor_candidate": 13, "walking_evidence_pending_not_replaced_by_driving": 1, "required_poi_needs_ferry_evidence": 1}。

接受数据城市分布：{"三亚": 2, "成都": 2, "上海": 2, "南京": 4, "苏州": 2}。

接受数据天数分布：{"2": 7, "3": 3, "1": 2}。

当前状态分类：{"accepted": 12, "api_or_schema_failed": 2}。

异常熔断：[]。

当前版本允许最多270次调用，加上前序30次最多300次；全局累计最多916次，硬上限1000。SDK关闭隐式重试，最多44次本地规则返工；报告输入超过20000 Token会告警，失败输出超过20000或任意输出超过40000时停止排队，本版本报告总Token达到250万也停止排队（在途请求可能有尾量）。所有失败同样记账。

九条相同生产提示词下的新教师原始答案已复用并重审，因此记录数不等于本版本新调用数；复用来源和旧规则误判保存在账本中，原始文本不改写。

此前接口将结构化格式返回成普通文本，并在禁用工具时返回web_search_call；现已将格式与不联网要求补入真实生产提示词，并继续检查原始响应和异常用量。服务端是否严格执行仍以每次响应为准，不能仅凭请求参数保证。

训练候选每条至少15个。西岛/蜈支洲缺渡船证据时从可选池排除并以真实同城点补足；必去该点的输入隔离。没有用驾车数据代替步行或船班。

接受表示通过确定性路线、已覆盖的事实/数值检查、严格Schema和3584长度检查，不等于完整人工语义认证。检查还包括按天范围识别的时长断言，避免把首日120/次日90误作全程矛盾。

完成后仅导出preview和复核明细；需继续审阅分布、解释、错误拒绝与复杂交通，才发布训练集。冻结50题不被改写；后续模型比较必须说明当前生产提示词升级，必要时为所有模型增加同协议配对比较。

参考：[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。

## 2026-10-25 v2 大候选池教师集与 SFT

- 语义抽查：37 条分层样本；154 条 accepted 中 150 条 semantic_pass、4 条 semantic_minor_issue、0 条 semantic_reject。轻微项仅涉及解释中的主题/偏好泛化表述，不淘汰路线。
- 最终候选：`data/post_train/v2_large_pool_teacher_contract_v3/sft_candidate/`，154 条 SFT，0 条 DPO；消息直接来自生产捕获，未复制旧答案。
- 服务器 `/home/jiale/TravelMind-posttrain` 已完成 SFT smoke（5 steps）与全量 SFT（154 条、3 epochs、30 steps），GPU0 峰值 21.64 GiB allocated / 22.98 GiB reserved，无 OOM；最终 train loss 0.7821。
- 合并模型：`/home/jiale/TravelMind-posttrain/models/TravelMind-Planner-7B-SFT-v2-large`，当前 vLLM 服务模型名 `TravelMind-Planner-SFT-v2-large`。
- 独立 heldout 评测实际包含 24 例：Planner-Only JSON 100%、硬约束通过 79.2%、软 rubric 100%、任务成功 79.2%、候选越界 4.2%；Rules+Rework JSON 100%、硬约束通过 91.7%、软 rubric 95.8%、任务成功 91.7%、候选越界 0%、平均返工 0.17。
- 当前结论：SFT 已显著降低候选越界并提高规则闭环成功率，但仍有 2 个失败样本（1 个开放窗口返工后结构化失败，1 个重复景点/道路证据不足）；DPO 暂不启动，先做失败归因与是否需要补充偏好对。
