# SFT-v2开发验证

固定30条开发验证、零返工、相同2048输出预算；Base和SFT-v1沿用已保存原始请求响应。
本表不是独立测试或四组最终结果，所有失败均计入分母。

|模型|完整约束通过|JSON解析|总Token|
|---|---:|---:|---:|
|Base|0/30|30/30|47519|
|SFT-v1|18/30|29/30|49081|
|SFT-v2|17/30|30/30|47611|

相对SFT-v1改善4例，退化5例。
须检查配对失败、截断和解释质量，再决定DPO。既有errors字段混合请求/解析/评分失败，不能统称接口调用错误。
SFT-v2错误类别：{"heat_exposure_or_unknown_indoor": 3, "duplicate_poi": 5, "commute_time_conflict": 1, "route_structure": 3, "opening_time_conflict": 2, "road_unverified": 1, "indoor_constraint": 1, "weather_outdoor_conflict": 1}
