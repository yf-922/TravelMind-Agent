# 仅迁移提示词对照实验

同30例、三种子、temperature0.3、输出2048；v1/v2复用逐字一致的原始请求记录，v3新运行。未使用独立测试集。

|模型|各seed通过 /30|规则通过 /90|三次全过 /30|规则+解释 /90|
|---|---|---:|---:|---:|
|v1|[18, 16, 16]|50|11|47|
|v2|[17, 20, 20]|57|14|49|
|v3|[17, 17, 17]|51|10|43|
|mc|[17, 19, 21]|57|14|48|

mc使用v3中完全相同的249条历史输入与答案，不加27条补数，固定48步。保留4条既有开始端点措辞问题作为诊断控制，不能视为部署数据认证。固定步数下曝光频率和epoch不同；不能仅据本实验断言补数本身有害。

mc错误：{"duplicate_poi": 7, "route_structure": 7, "commute_time_conflict": 4, "indoor_constraint": 5, "weather_outdoor_conflict": 5, "heat_exposure_or_unknown_indoor": 10, "opening_time_conflict": 10, "road_unverified": 2, "visit_too_short": 2, "pace_count": 1}

没有自动启动DPO；先检查逐例错误和解释再决定。
