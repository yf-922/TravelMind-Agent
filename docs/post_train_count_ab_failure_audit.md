# B 组失败审计

审计对象：`data/post_train/count_infeasible_ab_20261025/evaluation_total96/regression/B_*`。

## 结论

B 组旧回归集的 9 条“解析失败”并不是服务错误、超时或 JSON 解码错误。全部异常均为本地 rubric 在解析后的时间字段上抛出的 `ValueError: invalid clock`。

受影响的唯一种子组合为：

|模式|题目|种子|类型|城市|含义|
|---|---|---:|---|---|---|
|Planner-Only|representative50-1-04|43|multi_day_allocation|南京|两日 1+2 个景点|
|Planner-Only|representative50-2-04|43|multi_day_allocation|上海|两日 1+2 个景点|
|Planner-Only|representative50-4-03|43|latest_instruction|成都|最新数量指令|
|Planner-Only|representative50-5-08|44|midday_break|苏州|午间禁行窗口|
|Rules+Rework|以上4条|同上|同上|同上|返工模式首轮同样触发|
|Rules+Rework|representative50-4-08|43|midday_break|成都|午间禁行窗口|

## 证据解释

每条失败记录都有非零 `usage_total`，没有 `HTTPError`、超时或进程错误。相同题目的其他种子通常能正常通过，说明不是题目输入损坏或模型服务整体故障，而是特定种子生成了不符合 `HH:MM` 的时间字符串。`TravelRoute` 的 Pydantic schema 允许普通字符串，因此评测脚本在调用 `score_case()` 的 `minutes()` 时才抛出 `invalid clock`；异常发生在 `parse_success=True` 写入之前，导致当前结果把“JSON 可解析但时间格式不合法”记成了 parse failure。

因此这 9 条不能删除或改判为成功：它们确实暴露了模型输出时间格式的鲁棒性问题。但它们应单独计为 `schema_semantic_time_format`，不应与 vLLM/JSON 解析失败混为一类。

## 对 B 结论的影响

按当前严格口径，B 的旧集结果为 Planner-Only 125/150、Rules+Rework 138/150，并将 4/5 条时间格式失败计入分母。若只修正统计字段而不改变判定，这些分数不变；不能把它们补成通过。B 的 Rules+Rework 仍明显优于 A（125/150），但正式采用前必须解决时间格式约束或在结构化输出层增加严格 `HH:MM` 校验。

本审计没有重新调用付费 API，也没有修改模型权重或原始评测结果。
