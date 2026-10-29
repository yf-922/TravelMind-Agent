# 当前生产 Schema：Base、SFT、DPO 50×3 对照

四组补测已于北京时间2026-10-27 13:00左右完成，耗时约63分40秒。vLLM关闭已验证，GPU0显存15 MiB。该报告取代旧Schema跨批对比的效果结论。

主要结论：Base规则首轮103/150、闭环115/150；B-SFT为122/150、142/150；B-DPO为132/150、142/150。因此B-DPO的可见收益在首轮（+10条，+6.67个百分点），闭环总分与B-SFT持平。规则+解释闭环同为138/150。B-DPO相对Base首轮+19.33个百分点，闭环+18个百分点，但这包括整个SFT+DPO路径的收益。

B闭环Token从603952降到595410（约1.4%），平均返工0.26降至0.24，时延17.24秒至18.15秒；没有显著成本优势。v2-DPO相对v2-SFT各模式仅增1条、无解仍0/15，规则+解释未改善；A-DPO闭环增7条，无解10/15至15/15，但仍落后B-SFT。

同题配对后B首轮18条改善、8条退步；闭环4条改善、4条退步。案例bootstrap首轮95%区间约0至13.33个百分点，处于零边界；不能凭本开发集宣布稳定泛化收益。优先在冻结留出集上比较B-SFT/B-DPO，不再按本50题继续补数据。时延受LoRA服务方式及跨批运行影响，不能直接归因于模型计算量变化。

50题开发集×seed42/43/44；全部已完成，共2100条。逐题messages、当前Schema、system、温度0.3、max_tokens2048均核验一致；上下文8192、并发4。无付费教师API。
三组DPO来自此前完整运行，Base和三组SFT来自本次补测。各组解析150/150且无异常；时间格式受结构化生成约束，不能据此证明模型自行学会合法时间。该50题已参与多轮诊断，不是全新独立测试集。

“规则通过”是硬约束评分；本数据没有独立的常识评分器，因此不把它另命名为Commonsense。酒店/航班/工具幻觉也没有对应测试字段。
|版本|模式|规则 /150|规则+解释 /150|常规 /120|困难 /30|有解 /135|无解 /15|Token|平均秒|平均返工|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|base-sft|planner-only|103|103|84|19|96|7|472590|6.98|0.00|
|base-sft|rules-rework|115|115|95|20|107|8|841804|12.01|0.81|
|v2-sft|planner-only|111|111|97|14|111|0|470325|6.55|0.00|
|v2-sft|rules-rework|120|120|105|15|120|0|769869|10.39|0.68|
|v2-dpo|planner-only|112|111|98|14|112|0|471189|14.31|0.00|
|v2-dpo|rules-rework|121|119|106|15|121|0|781639|23.47|0.70|
|A-sft|planner-only|112|112|86|26|101|11|463225|11.97|0.00|
|A-sft|rules-rework|122|122|97|25|112|10|787381|18.39|0.65|
|A-dpo|planner-only|116|114|88|28|103|13|463126|12.32|0.00|
|A-dpo|rules-rework|129|128|99|30|114|15|709414|17.33|0.49|
|B-sft|planner-only|122|119|97|25|112|10|470707|14.13|0.00|
|B-sft|rules-rework|142|138|112|30|127|15|603952|17.24|0.26|
|B-dpo|planner-only|132|129|103|29|118|14|471372|14.61|0.00|
|B-dpo|rules-rework|142|138|112|30|127|15|595410|18.15|0.24|

配对改善/退步及按案例分层bootstrap区间：

```json
{
  "v2/planner-only": {
    "improved": 5,
    "regressed": 4,
    "delta_pp": 0.6666666666666667,
    "case_bootstrap_95pct": [
      -3.3333333333333335,
      4.666666666666666
    ],
    "regressed_cases": [
      "representative50-1-06",
      "representative50-3-06",
      "representative50-4-05",
      "representative50-5-07"
    ]
  },
  "v2/rules-rework": {
    "improved": 3,
    "regressed": 2,
    "delta_pp": 0.6666666666666667,
    "case_bootstrap_95pct": [
      -1.3333333333333333,
      2.6666666666666665
    ],
    "regressed_cases": [
      "representative50-1-06",
      "representative50-1-08"
    ]
  },
  "A/planner-only": {
    "improved": 13,
    "regressed": 9,
    "delta_pp": 2.666666666666667,
    "case_bootstrap_95pct": [
      -6.666666666666667,
      12.0
    ],
    "regressed_cases": [
      "representative50-1-01",
      "representative50-2-01",
      "representative50-2-06",
      "representative50-4-05",
      "representative50-5-01"
    ]
  },
  "A/rules-rework": {
    "improved": 10,
    "regressed": 3,
    "delta_pp": 4.666666666666667,
    "case_bootstrap_95pct": [
      -1.3333333333333333,
      11.333333333333334
    ],
    "regressed_cases": [
      "representative50-2-02",
      "representative50-4-05"
    ]
  },
  "B/planner-only": {
    "improved": 18,
    "regressed": 8,
    "delta_pp": 6.666666666666667,
    "case_bootstrap_95pct": [
      1.1102230246251565e-16,
      13.333333333333334
    ],
    "regressed_cases": [
      "representative50-1-06",
      "representative50-1-08",
      "representative50-2-04",
      "representative50-2-05",
      "representative50-2-07",
      "representative50-3-06",
      "representative50-4-06",
      "representative50-5-07"
    ]
  },
  "B/rules-rework": {
    "improved": 4,
    "regressed": 4,
    "delta_pp": 0.0,
    "case_bootstrap_95pct": [
      -3.3333333333333335,
      3.3333333333333335
    ],
    "regressed_cases": [
      "representative50-1-08",
      "representative50-3-05",
      "representative50-4-03",
      "representative50-5-06"
    ]
  }
}
```

错误分类与源文件见 comparison.json。时延为不同时间段、并发4的客户端响应时间，不能当作严格硬件隔离性能实验。偏好验证准确率不等于路线任务通过率。
