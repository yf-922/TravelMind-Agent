# 当前生产协议重训后的统一评测启动记录

2026-10-25 16:07（北京时间）在本机启动后台队列，模型推理由服务器GPU0执行。四组模型：Base、v1-current、v2-current、v2-large。前两版current模型使用本次重新生成的生产协议教师答案训练；v2-large为已有版本。

## 固定设置

- 五城市50题，40条常规、10条困难；每题seed=42/43/44。
- 每个模型分别评测Planner-Only和Rules+Rework，150条/模式，共1200条最终结果，返工调用另计。
- 所有模型使用当前生产`capture_messages(state)`及相同`TravelRoute` JSON Schema；system SHA256为`402a279c8e218de671211b5c820b88d6cedb181beea8fd92bb7aef2fe4e35c2c`。
- temperature=0.3，max_tokens=2048，上下文8192，并发4；输入最长3301 Token，避免使用4096上下文挤压输出预算。
- Planner-Only仅生成一次。Rules+Rework最多初稿+3次规则反馈重排；解释一致性单独统计，不触发返工。不调用LLM Reviewer或付费教师API。
- 模式分别生成初稿；同题同seed构成配对，不宣称vLLM各次初稿逐字相同。

## 启动核查

冻结资产和生产消息核对通过，四组模型的配置与权重分片齐备；相关`tests/test_representative_eval50.py`测试7项通过。旧的停止实验目录没有继续运行，新结果单独保存在`current_retrained_20261025`。

启动器为`scripts/evaluate_current_protocol_sft.py`；本机后台进程PID=15832，日志为`current_protocol_eval.stdout.log`与`current_protocol_eval.stderr.log`。本机需要保持开机、网络连接并关闭自动休眠，以便本机评分队列持续请求服务器。

16:09初步核查：Base Planner-Only已完成80/150，全部解析，未见接口或执行错误，均值约6.9秒/条。初估整个队列40–70分钟，返工比例会影响耗时；需在返工模式开始后校正。

稳定性核查通过后停止人工监视：Base Planner-Only完成150/150，均值6.86秒/条；Rules+Rework完成17条、31轮生成，其中6条发生返工，均值11.68秒/条。两种模式均无接口或执行错误、解析全部成功，原始每轮结果已正常落盘。后台继续执行，校正后的预计完成时间为北京时间17:00–17:20，返工比例和模型切换可能使时间变化。此时167/1200条最终记录已写入，`status.json`的150只反映完整结束的模式。

## 输出及解释

输出目录：`data/post_train/representative_eval50_v1/current_retrained_20261025/`。

- `binding.json`：题集、模型分片索引、提示词、评分代码、参数的绑定。
- `status.json`：模型和模式阶段；`done`只计已经完整结束的模式，实时条数以各模式`results.jsonl`为准。
- `{模型}_{模式}/results.jsonl`：每轮请求、输出、规则与解释错误、Token和耗时，支持断点续跑。
- `summaries.json`：分层、城市、规则及解释故障汇总。
- `paired_comparisons.json`：按案例分层bootstrap；同题3个seed作为相关重复，避免当作150道独立题。
- `report.md`和`docs/post_train_current_protocol_sft_evaluation.md`：全部结束后的统一对照表。

完成或失败后恢复Base服务。接口错误会在模式结束时阻止后续模型评测；模型自身的输出解析或截断问题保留为失败结果。既有50题已被反复用于诊断，报告明确标注开发基准，不能替代新独立测试。历史API教师指标只有消息与评分口径一致才可直接比较，本轮不自动混入异协议历史分数。
