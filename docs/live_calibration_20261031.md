# 候选补查、时间轴与早停补丁：真实联调记录

> 本文保留前一阶段历史结果。总时间窗口、官方名称匹配补丁及后续真实复跑结果见 [最终补丁验收](live_calibration_final_20261031.md)。

## 执行口径

2026-10-31 UTC 执行，真实 `openai-proxy / gpt-6.1-sol` 与高德 API，主生产 LangGraph，不用离线 POI 替代真实回包。四条原始请求不改写为更容易通过的请求。每请求最多返工3次、最多4份正常快照，补查最多2轮；HTTP预算为 `min(10, max(4, requested_names数量+2))`。

Redis仅在评测子进程禁用，不修改环境文件。脚本不自动重跑整单，生产内部的解析/接口重试仍有效。这里报告单次端到端观测，不是P50或稳定性能收益。原始日志位于被Git忽略的 `data/live_calibration/`，未包含密钥。

## 生产改动

- `quality.py`：景点驻留至少30分钟、餐饮至少20分钟，非法时间、零时长、倒挂、原顺序重叠均进入L1 `invalid_time_slot`；新增餐饮与景点之间已核验路段的通勤时间冲突检查。
- `candidate_react.py`：动态HTTP预算；缺失点名优先排队精确查询，再去通用后缀查核心词；原名仍用于回包身份校验，不因扩大关键词而接纳相似但无关地点。两次正常无匹配才剔除；接口失败仍是未核实。
- `schemas.py`：补查动作携带原始 `requested_name`，区分原名和核心词。
- `nodes.py`：L0/L1全过且枚举为HIGH/MEDIUM与GOOD/FAIR即早停，模型的非严重否决保留提醒；LOW/POOR或硬问题继续返工。快照上限及原有结构化失败终止标记均不能再次打开Planner回环。
- `prompts.py`：同步时间轴与最短驻留约束。`tests/test_live_calibration_patches.py`覆盖新增行为，既有回归替身同步新路由/查询语义。
- `scripts/evaluate_live_calibration.py`：保存完整逐例结果、阶段状态、快照、查询轨迹和用量归因，支持 `--cases` 明确选取复跑对象。

## 最新可用真实观测

| Case | 状态 | 快照数/返工数 | 选中轮次 | 补查轮次 | HTTP使用/上限 | 端到端毫秒 |
|---|---|---:|---:|---:|---:|---:|
| 1 南京两日历史慢游、避商业街 | SUCCESS_WITH_WARNINGS | 1/0 | 1 | 2 | 4/4 | 119876.2 |
| 3 重庆五处点名、单段步行≤2公里 | PARTIAL_SUCCESS | 4/3 | 3 | 2 | 7/7 | 420873.0 |
| 5 南京四小时、三处均必去 | SUCCESS_WITH_WARNINGS | 1/0 | 1 | 2 | 4/5 | 135432.3 |
| 6 上海唯一虚构点名 | FAILED | 0/0 | 无 | 2 | 2/4 | 31990.5 |

这里的“补查轮次”为 `candidate_search_round`，HTTP使用为候选检索预算计数，不包含天气/餐馆/道路API，不代表全部外部请求数。轮次不等于每个目标都进行了两次HTTP。Case6两种词的HTTP均在第一轮搜索阶段执行完，validator之后仍推进到补查轮次2。

Case1来源：`20261031T143012Z/results.json`；Case3来源：`20261031T142617Z/results.json`；Case5/6最终复跑来源：`20261031T143640Z/results.json`。首批 Case5/6 仍保留在 `20261031T141817Z`，但最终表采用补丁后的明确复跑，不用旧记录替代。

## 快照明细

六维顺序：named_poi / pace / continuity / weather / preference_match / diversity_and_dining。

| Case/iteration | L0数 | L1数与代码 | soft_score | 六维得分 | Reviewer枚举 |
|---|---:|---|---:|---|---|
| 1/1 | 0 | 0 | 100 | 20/20/15/15/25/5 | HIGH/GOOD |
| 3/1 | 0 | 4：2 commute_time_conflict，2 walking_constraint | 72 | 20/15/10/10/15/2 | MEDIUM/FAIR |
| 3/2 | 0 | 4：2 commute_time_conflict，2 walking_constraint | 72 | 20/15/10/10/15/2 | MEDIUM/FAIR |
| 3/3 | 0 | 4：2 commute_time_conflict，2 walking_constraint | 82 | 20/15/10/10/25/2 | HIGH/FAIR |
| 3/4 | 0 | 4：2 commute_time_conflict，2 walking_constraint | 72 | 20/15/10/10/15/2 | MEDIUM/FAIR |
| 5/1 | 0 | 0 | 77 | 20/15/10/15/15/2 | MEDIUM/FAIR |
| 6 | 无快照 | 前置失败 | 不评分 | 不评分 | 未调用Reviewer |

Case1真实Reviewer approved=True；本次实跑不能单独证明覆盖 approved=False 的早停分支，该分支由HIGH/MEDIUM×GOOD/FAIR四组合单测验证。首批Case1 approved=False时已早停，但其餐饮通勤检查不完整，不作为最终正确路线的证据。

## 已验证与未达标项

1. **时间轴拦截已通过受控测试，Case5真实复跑没有命中零时长/倒序。** 最终景点时间为08:30–09:00、09:40–10:10、10:50–11:20，均达到30分钟且有序，故不能人为改成PARTIAL_SUCCESS。模型把4小时窗口超限（午餐计入约4小时39分、晚餐在窗口外）作为提醒返回；当前仍缺少覆盖餐饮在内的总窗口确定性检查，该条不能称为“用户全部需求已满足”。
2. **动态预算保障查询机会，不等于召回率100%。** Case3五个目标5/5均获得精确HTTP查询，预算由4扩为7；最终仅3/5目标匹配入池：磁器口、洪崖洞、三峡博物馆。解放碑和南山一棵树的名称/类型身份匹配与接口异常仍未解决；两次查询无被接纳匹配并不证明现实地标不存在。日志中的“未找到指定景点解放碑”是实际系统文案，不是本报告的事实结论。
3. **早停与历史择优都生效。** Case1一版100分后停止；Case3四版均有L1，按冲突数再按分数选第3版82分，不是最后一版。
4. **核心词补查与前置失败生效。** 最终复跑 Case6 依次查询“星河幻境博物馆”和“星河幻境”，两次均收到25条但全部身份不匹配的回包，随后确认 `named_poi_unavailable`，days为空且Planner快照为0；不是把接口异常误判为不存在。受控单测另覆盖接口失败时的 `named_poi_unverified`。
5. **诊断尚有误差。** 每变体HTTP小预算耗尽会把一次接口错误最终记成 `BUDGET_EXHAUSTED`，不一定表示全局预算用尽。步行限制当前使道路节点选择walk；Planner写“打车”不能自动得到drive核验，景点直连规则也与包含餐馆的道路链不完全一致。Case3因此不能宣称交通可执行性已解决。

## 真实输出文案举例

- Case1：`Day2 晚餐位于夫子庙片区，可直接前往餐厅用餐，无需增加周边商业街游览。`
- Case3：`Day1 景点与餐饮衔接时间不足：北步园火锅村(总店)→磁器口古镇，预留 105 分钟，已核验通勤需 172 分钟`；`未核验步行路段：重庆中国三峡博物馆→磁器口古镇`。failure_reasons为空，具体冲突保留为warnings。
- Case5：`Day1只有4小时，建议在12:30结束全部主行程；12:40的午餐和17:00的晚餐可保留为自行选择的餐厅推荐，不必作为当天必走安排。` 注意这是提醒，不代表JSON已移除两餐。
- Case6：`{"code":"named_poi_unverified","message":"暂未能核实用户唯一指定景点：星河幻境博物馆，请稍后重试","stage":"PRECHECK"}`。

## 全部执行记录，不隐藏失败

| 批次UTC | Case | 耗时毫秒 | 备注 |
|---|---|---:|---|
| 20261031T141817Z | 1 | 118134.7 | 首批；发现餐饮通勤漏检，不能作为最终物理正确性证据 |
| 20261031T141817Z | 3 | 353038.5 | 首批；餐饮通勤补丁前 |
| 20261031T141817Z | 5 | 164875.5 | 首批；保留但不作为最终复跑 |
| 20261031T141817Z | 6 | 20560.8 | 首批；核心词接口错误，未核实 |
| 20261031T142356Z | 1 | 303234.3 | 4次结构化输出失败后错误重入，出现iteration5；已据此修补终止路由，不计为上限验收通过 |
| 20261031T142617Z | 3 | 420873.0 | 增加餐饮通勤检查后的明确复跑；4快照，选3 |
| 20261031T143012Z | 1 | 119876.2 | 终止路由补丁后明确复跑；1快照 |
| 20261031T143640Z | 5 | 135432.3 | 补丁后明确复跑；1快照 |
| 20261031T143640Z | 6 | 31990.5 | 补丁后明确复跑；双词均无身份匹配，前置失败 |

上述共9次整单运行、监控记录60次LLM调用，其中4次失败。所有调用的Token均为监控估算（usage_exact_ratio=0），没有供应商精确usage，费用为unpriced，不把默认estimated_cost_usd=0解释为免费。没有据这些单次观测生成平均提速或精确费用指标。

## 验证

- `python -m pytest -q`：**839 passed，3 warnings，52.04s**。警告为第三方弃用提示。
- `python scripts/validate_eval_assets.py`：通过，未发外部请求。
- `python scripts/evaluate_candidate_react.py`：固定响应非边界20/23=86.96%；点名9/9、室内7/9、数量4/5；预算合规28/28。此结果不是本页四条真实请求的成功率。
- `git diff --check`：通过。

复现真实联调（会产生外部调用，不是pytest的一部分）：

```powershell
python scripts/evaluate_live_calibration.py --concurrency 2
python scripts/evaluate_live_calibration.py --concurrency 1 --cases case1_nanjing_slow_history
python scripts/evaluate_live_calibration.py --concurrency 1 --cases case3_chongqing_dense_cross_area --timeout 600
```

默认外层超时360秒，重庆明确复跑使用600秒外层限时；不增加节点返工或HTTP预算。真实天气与POI会变化，不能承诺再次运行得到完全相同快照和耗时。结论是补丁机制已实现并有回归覆盖，但四个真实场景并未全部达到预期验收，不作为简历100%成功率数据。
