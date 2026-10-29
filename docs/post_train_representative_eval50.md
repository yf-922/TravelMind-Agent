# 五城市50例开发评测集 v1

用途：替代仅3–4候选的单一开发分布，用于模型选择与失败诊断。不是盲测独立集，不进入训练。旧30例继续作为稀疏压力回归，旧丽江/景德镇独立集保持不变。

## 分布与前置条件

|城市|常规|困难|常规候选数|
|---|---:|---:|---:|
|南京|8|2|23|
|上海|8|2|23|
|三亚|8|2|19|
|成都|8|2|22|
|苏州|8|2|24|

共40例常规前置、10例困难。常规不等于无约束：它要求真实候选充足并有可行路线，仍检验指令优先级、时间边界等。80/20是暂定设计权重，尚无生产流量统计支持；5城等权也不是实际城市流量比例。

## 约束类型

|每城编号|类型|主要陷阱|
|---|---|---|
|01|默认与用户上限|系统上限3，用户最多2，并明确要求本日2个；不能填默认上限，也不能空排|
|02|必去与排除|同时包含必去和禁去地点，要求逐字匹配真实候选|
|03|修改优先级|旧请求恰好3个；当前修改降到2个并更新必去/排除，不能照搬旧要求|
|04|多日非均匀分配|第一天1个、第二天2个；全程3个不同景点，不能误读成每天2个|
|05|未触发高温阈值|37.9°C且只允许中午至下午，不能无依据套用高温禁游|
|06|高温与未知属性|38°C、11:00–15:00禁未知/室外；11点结束、15点开始允许|
|07|严格时间端点|10:00之后意味着最早10:01；同时必须满足指定景点与开放/入场时间|
|08|休息区间与解释|12:00–14:00休息，端点允许；休息不等于应虚构餐馆|
|09|稀疏有解|3天3候选，每天1个；少排可接受，全程去重|
|10|客观无解|3城为2候选却要求3个不同地点；2城为高温仅限正午但无已认证室内候选；应空排并说明冲突|

地点、开放时间、道路通勤和解释一致性对所有可行题共同评分。无老人登阶、场所商业性等缺证据标签：未独立核实这些属性，不能凭名称判错。缺失天气、未知开放时间、道路故障等另属后续故障专项，本版不声称覆盖全部Agent故障。

## 证据与审计

- 新检索26次逻辑调用（HTTP重试可能增加），原始响应与过滤账本保存。5城候选原始池24–25个，人工复核后常规池19–24个。按已知评分4.5阈值过滤，缺评分不自动排除。
- POI子点/设施、商店误检、岛屿渡船等排除及理由见manual_candidate_review.json。仍不是实地完整实体核验。
- 保存122次道路批量调用的账本与响应，共2874条有向驾车记录（含复核排除前的额外记录）；最终保留池所有有向点对证据齐全。驾车记录不包含停车、排队、未来交通或渡船保证。
- 开放时间保留原文，并取可解析时间区间的保守交集，统一为2026-11-04至11-06周三至周五的输入快照。不是未来营业承诺；复杂季节、节假日与语义解析仍有限制，原文可回查。
- 所有新POI室内属性为unknown；不从博物馆名称升级为室内。高温题通过调整时段可行；无解题明确提供未知属性条件。
- 45例可行题均有通过评分的路线见证，5例无解题有集合/数量冲突证明；合计50份正控制及155份负控制通过审计。见证只用于验证题目，不作为模型输入或训练数据。
- 实际生产Joint Planner messages直接捕获并冻结；rubric、见证和不可行证明不会发给模型。正常请求无新增知识检索证据，餐馆不在本轮范围。
- Qwen原始tokenizer检查：输入1076–3010 Token；加2048输出最多5058，运行服务需至少8192上下文，禁止截断输入。

## 结果怎么看

分别报告常规40例成功率、困难10例成功率；再拆45例可行路线与5例合理拒绝。每城、每约束类型列失败，避免总分掩盖错误。共同通过要求规则与解释都通过；解释检查为已实现的确定性核对，不是完整语义认证。

同时统计重复/越界、开放/最晚入场、通勤、高温、每日数量、未请求餐馆、平均景点数与空日、Token和延迟。空日只在无解题按正确拒绝评价，不能靠拒绝可行题提高成绩。

50条来自5城×10类，模板及地点存在相关性，不能视为50个完全独立任务分布。首轮建议固定3个种子，按案例聚合；重复生成150次不等于150道独立题。

## 运行与状态

已冻结FROZEN.json及数据/源代码哈希，评测入口会检查变更。新增7项单测及全量602项回归通过（3条既有弃用警告）；mock已验证50例流程，无模型调用、无付费教师Token。mock的50/50仅表示见证与评分器自检成功，不能作为模型效果。

```powershell
python -X utf8 scripts/evaluate_representative50.py --mock --seeds 42
# 先在GPU0加载指定模型，并将vLLM上下文设为8192；然后运行：
python -X utf8 scripts/evaluate_representative50.py --model-name TravelMind-Planner-7B-SFT-v2 --tag v2
```

入口固定使用本地服务器OpenAI兼容服务，无付费接口回退。当前仅实现Planner-Only；规则返工闭环及API教师的同协议评测需要后续单独执行并记录预算。本次未启动任何模型效果评测或训练。

## 全部案例

|ID|城市|分组|类型|候选数|预期|
|---|---|---|---|---:|---|
|representative50-1-01|南京|normal|upper_limit|23|feasible|
|representative50-1-02|南京|normal|required_excluded|23|feasible|
|representative50-1-03|南京|normal|latest_instruction|23|feasible|
|representative50-1-04|南京|normal|multi_day_allocation|23|feasible|
|representative50-1-05|南京|normal|below_heat_threshold|23|feasible|
|representative50-1-06|南京|normal|heat_boundary|23|feasible|
|representative50-1-07|南京|normal|strict_start|23|feasible|
|representative50-1-08|南京|normal|midday_break|23|feasible|
|representative50-1-09|南京|hard|sparse_feasible|3|feasible|
|representative50-1-10|南京|hard|infeasible|2|explained_infeasible|
|representative50-2-01|上海|normal|upper_limit|23|feasible|
|representative50-2-02|上海|normal|required_excluded|23|feasible|
|representative50-2-03|上海|normal|latest_instruction|23|feasible|
|representative50-2-04|上海|normal|multi_day_allocation|23|feasible|
|representative50-2-05|上海|normal|below_heat_threshold|23|feasible|
|representative50-2-06|上海|normal|heat_boundary|23|feasible|
|representative50-2-07|上海|normal|strict_start|23|feasible|
|representative50-2-08|上海|normal|midday_break|23|feasible|
|representative50-2-09|上海|hard|sparse_feasible|3|feasible|
|representative50-2-10|上海|hard|infeasible|23|explained_infeasible|
|representative50-3-01|三亚|normal|upper_limit|19|feasible|
|representative50-3-02|三亚|normal|required_excluded|19|feasible|
|representative50-3-03|三亚|normal|latest_instruction|19|feasible|
|representative50-3-04|三亚|normal|multi_day_allocation|19|feasible|
|representative50-3-05|三亚|normal|below_heat_threshold|19|feasible|
|representative50-3-06|三亚|normal|heat_boundary|19|feasible|
|representative50-3-07|三亚|normal|strict_start|19|feasible|
|representative50-3-08|三亚|normal|midday_break|19|feasible|
|representative50-3-09|三亚|hard|sparse_feasible|3|feasible|
|representative50-3-10|三亚|hard|infeasible|2|explained_infeasible|
|representative50-4-01|成都|normal|upper_limit|22|feasible|
|representative50-4-02|成都|normal|required_excluded|22|feasible|
|representative50-4-03|成都|normal|latest_instruction|22|feasible|
|representative50-4-04|成都|normal|multi_day_allocation|22|feasible|
|representative50-4-05|成都|normal|below_heat_threshold|22|feasible|
|representative50-4-06|成都|normal|heat_boundary|22|feasible|
|representative50-4-07|成都|normal|strict_start|22|feasible|
|representative50-4-08|成都|normal|midday_break|22|feasible|
|representative50-4-09|成都|hard|sparse_feasible|3|feasible|
|representative50-4-10|成都|hard|infeasible|22|explained_infeasible|
|representative50-5-01|苏州|normal|upper_limit|24|feasible|
|representative50-5-02|苏州|normal|required_excluded|24|feasible|
|representative50-5-03|苏州|normal|latest_instruction|24|feasible|
|representative50-5-04|苏州|normal|multi_day_allocation|24|feasible|
|representative50-5-05|苏州|normal|below_heat_threshold|24|feasible|
|representative50-5-06|苏州|normal|heat_boundary|24|feasible|
|representative50-5-07|苏州|normal|strict_start|24|feasible|
|representative50-5-08|苏州|normal|midday_break|24|feasible|
|representative50-5-09|苏州|hard|sparse_feasible|3|feasible|
|representative50-5-10|苏州|hard|infeasible|2|explained_infeasible|
