# SFT-v3统一协议开发集对比

同30例、三种子、temperature0.3、输出2048；v1/v2复用逐字一致的原始请求记录，v3新运行。未使用独立测试集。

|模型|各seed通过 /30|规则通过 /90|三次全过 /30|规则+解释 /90|
|---|---|---:|---:|---:|
|v1|[18, 16, 16]|50|11|47|
|v2|[17, 20, 20]|57|14|49|
|v3|[17, 17, 17]|51|10|43|

训练使用276条（249条审计迁移+27条新捕获），固定48步；改变数据与输入协议，不能作单因素因果结论。

v3错误：{"heat_exposure_or_unknown_indoor": 10, "route_structure": 10, "duplicate_poi": 10, "indoor_constraint": 5, "weather_outdoor_conflict": 5, "commute_gap": 1, "commute_time_conflict": 2, "opening_time_conflict": 8, "visit_too_short": 2}

没有自动启动DPO；先检查逐例错误和解释再决定。
