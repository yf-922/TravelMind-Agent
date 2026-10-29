# 统一约束协议：完整开发集重复比较与定向补数

生成时间：2026-10-24T19:22:28

30个固定开发状态，每模型三次生成，temperature=0.3，输出上限2048，种子42/43/44。输入为重新捕获的新生产协议，不能与旧单次通过数直接作同协议比较。未使用独立测试集；重复生成并非90个独立问题。

| 模型 | seed42 /30 | seed43 /30 | seed44 /30 | 合计 /90 | 三次全过 /30 | 解析 /90 | 规则+解释检查 /90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 18 | 16 | 16 | 50 | 11 | 90 | 47 |
| v2 | 17 | 20 | 20 | 57 | 14 | 90 | 49 |

按每例三次成功数比较：v2改善8例，退步5例。仅代表本开发集，不能据此宣称独立泛化或因果归因。

## 错误分布

v1：`{"heat_exposure_or_unknown_indoor": 7, "commute_time_conflict": 6, "route_structure": 16, "opening_time_conflict": 8, "duplicate_poi": 8, "indoor_constraint": 6, "weather_outdoor_conflict": 6, "road_unverified": 5, "required_count": 4, "visit_too_short": 1}`
v2：`{"commute_time_conflict": 6, "duplicate_poi": 13, "heat_exposure_or_unknown_indoor": 6, "route_structure": 13, "opening_time_conflict": 4, "road_unverified": 1, "visit_too_short": 1, "commute_gap": 2, "indoor_constraint": 1, "weather_outdoor_conflict": 1}`

## 补数与清理

32个输入完成32个，合格32个，失败/隔离0个。正式调用25次，包含停止的pilot累计started 44次。已记账token 180792，中断未返usage部分不能假定为零。

合格样本训练token最大2133，未允许截断。已逐条重放确定性审计；这些检查不等于完整语义认证。

已确认的旧返工叙述synth-0127从新清理副本隔离。旧release_v1/v2及其原始messages、权重保留。未启动新SFT或DPO。新补数按新协议单独导出；历史数据若迁移新Prompt需单独审计并注明，不可静默改写。

## 下一步决策

根据三次全过数和同例成败波动优先判断稳定性，再检查重复、数量、高温和道路的成对变化。先做合格补数的语义审阅及新协议训练集版本冻结，再决定是否开展新SFT；DPO应在SFT稳定提升后另行决定。最终四组含API比较仍在封存独立集上统一记录。
