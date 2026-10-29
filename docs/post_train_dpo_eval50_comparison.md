# DPO 与对应 SFT：50×3 对照

本次三组两模式全部完成（900条最终记录），耗时约67分44秒。结束后已核查vLLM关闭，GPU0显存15 MiB。

主要观察：B-DPO在本次DPO三组中规则通过最多（首轮132/150、闭环142/150）；A-DPO首轮116/150、闭环129/150；v2-DPO首轮112/150、闭环121/150。v2的无解题仍为0/15，DPO没有解决该能力缺口。B闭环相较历史SFT仅多4条，且解释共同通过138/150，仍有输出解释问题。

所有SFT/DPO配对差值的案例bootstrap区间均跨0；叠加Schema差异，不能声称已证明DPO显著改善。建议保持训练数据不变，先同Schema补跑三个SFT对照，再在未用于调参的留出集验证。50题是开发基准，不得称为全新独立集。

50题为已多次用于诊断的开发集，不是全新独立测试。三种子为同题重复，不能视为150道独立题。SFT使用历史已完成记录，DPO使用本次记录；system、逐题首轮messages哈希、seed、温度、输出上限一致。服务上下文8192。未重新调用教师API。
重要混杂：DPO使用带严格HH:MM pattern的Schema（ae669174…），历史SFT没有该pattern（8e12d928…）。已验证当前Schema仅删除pattern即可复现旧哈希。时间格式及整体增益不能全部归因于DPO；需要SFT在同一Schema下重测。当前仅为描述性比较。

|版本|模式|规则 /150|规则+解释 /150|常规 /120|困难 /30|有解 /135|无解 /15|Token|平均秒|平均返工|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|v2-sft|planner-only|108|108|94|14|108|0|470390|6.66|0.00|
|v2-sft|rules-rework|118|118|103|15|118|0|776865|10.86|0.69|
|v2-dpo|planner-only|112|111|98|14|112|0|471189|14.31|0.00|
|v2-dpo|rules-rework|121|119|106|15|121|0|781639|23.47|0.70|
|A-sft|planner-only|107|107|82|25|97|10|463326|12.10|0.00|
|A-sft|rules-rework|125|124|98|27|113|12|778713|18.61|0.63|
|A-dpo|planner-only|116|114|88|28|103|13|463126|12.32|0.00|
|A-dpo|rules-rework|129|128|99|30|114|15|709414|17.33|0.49|
|B-sft|planner-only|125|124|100|25|115|10|470538|13.95|0.00|
|B-sft|rules-rework|138|135|108|30|123|15|598271|18.18|0.25|
|B-dpo|planner-only|132|129|103|29|118|14|471372|14.61|0.00|
|B-dpo|rules-rework|142|138|112|30|127|15|595410|18.15|0.24|

配对改善/退步及按案例分层bootstrap区间：

```json
{
  "v2/planner-only": {
    "improved": 7,
    "regressed": 3,
    "delta_pp": 2.666666666666667,
    "case_bootstrap_95pct": [
      -1.3333333333333333,
      7.333333333333333
    ],
    "regressed_cases": [
      "representative50-1-06",
      "representative50-4-05",
      "representative50-5-07"
    ]
  },
  "v2/rules-rework": {
    "improved": 4,
    "regressed": 1,
    "delta_pp": 2.0,
    "case_bootstrap_95pct": [
      -0.6666666666666666,
      6.0
    ],
    "regressed_cases": [
      "representative50-1-08"
    ]
  },
  "A/planner-only": {
    "improved": 15,
    "regressed": 6,
    "delta_pp": 6.0,
    "case_bootstrap_95pct": [
      -2.6666666666666665,
      15.333333333333336
    ],
    "regressed_cases": [
      "representative50-2-01",
      "representative50-2-06",
      "representative50-4-05"
    ]
  },
  "A/rules-rework": {
    "improved": 7,
    "regressed": 3,
    "delta_pp": 2.666666666666667,
    "case_bootstrap_95pct": [
      -2.6666666666666665,
      8.0
    ],
    "regressed_cases": [
      "representative50-2-02",
      "representative50-4-05"
    ]
  },
  "B/planner-only": {
    "improved": 14,
    "regressed": 7,
    "delta_pp": 4.666666666666667,
    "case_bootstrap_95pct": [
      -1.333333333333333,
      11.333333333333332
    ],
    "regressed_cases": [
      "representative50-1-06",
      "representative50-1-08",
      "representative50-2-04",
      "representative50-2-05",
      "representative50-3-06",
      "representative50-4-06",
      "representative50-5-07"
    ]
  },
  "B/rules-rework": {
    "improved": 8,
    "regressed": 4,
    "delta_pp": 2.666666666666667,
    "case_bootstrap_95pct": [
      -1.3333333333333333,
      6.666666666666667
    ],
    "regressed_cases": [
      "representative50-1-08",
      "representative50-3-05",
      "representative50-5-06"
    ]
  }
}
```

错误分类与源文件见 comparison.json。时延为不同时间段、并发4的客户端响应时间，不能当作严格硬件隔离性能实验。偏好验证准确率不等于路线任务通过率。
