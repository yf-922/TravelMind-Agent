# 清洗与补数SFT对照

> 复核更正：原解释检查漏掉selection_reasons中的时长矛盾。新增统一事后诊断后，v2/v4_clean/v4共同通过分别为99/75/83，而非仅凭原表认定v4提升。原始冻结成绩保持。详见[结果复核](post_train_quality_v4_result_review.md)。

冻结五城市50题×3种子，Planner-Only，不使用独立测试。v2复用同协议已保存结果。
训练从原始Qwen基座分别开始，48步、同LoRA/LR/种子；固定更新次数不代表Token或曝光完全相等。

|版本|规则 /150|规则+解释 /150|常规 /120|困难 /30|无解 /15|
|---|---:|---:|---:|---:|---:|
|v2|119|99|99|20|7|
|v4_clean|93|93|82|11|0|
|v4|108|108|96|12|0|

## 配对差值

{
  "v4_clean": {
    "rules": {
      "delta_percentage_points": -17.333333333333332,
      "paired_case_bootstrap_95pct": [
        -29.333333333333332,
        -5.333333333333333
      ],
      "note": "stratified case bootstrap, correlated seeds kept together; conditional on this designed development suite"
    },
    "joint": {
      "delta_percentage_points": -4.0,
      "paired_case_bootstrap_95pct": [
        -18.666666666666668,
        10.666666666666666
      ],
      "note": "stratified case bootstrap, correlated seeds kept together; conditional on this designed development suite"
    }
  },
  "v4": {
    "rules": {
      "delta_percentage_points": -7.333333333333332,
      "paired_case_bootstrap_95pct": [
        -18.0,
        4.0
      ],
      "note": "stratified case bootstrap, correlated seeds kept together; conditional on this designed development suite"
    },
    "joint": {
      "delta_percentage_points": 6.0,
      "paired_case_bootstrap_95pct": [
        -6.666666666666667,
        18.666666666666668
      ],
      "note": "stratified case bootstrap, correlated seeds kept together; conditional on this designed development suite"
    }
  },
  "augmented_vs_clean": {
    "rules": {
      "delta_percentage_points": 10.0,
      "paired_case_bootstrap_95pct": [
        0.6666666666666669,
        20.0
      ],
      "note": "stratified case bootstrap, correlated seeds kept together; conditional on this designed development suite"
    },
    "joint": {
      "delta_percentage_points": 10.0,
      "paired_case_bootstrap_95pct": [
        0.6666666666666669,
        20.0
      ],
      "note": "stratified case bootstrap, correlated seeds kept together; conditional on this designed development suite"
    }
  }
}

## 实际训练Token

{
  "v4_clean": {
    "samples": 736,
    "nonpadding_tokens": 1474124,
    "supervised_tokens": 271700
  },
  "v4": {
    "samples": 744,
    "nonpadding_tokens": 1576604,
    "supervised_tokens": 268808
  }
}

## 仅格式修复诊断（非主成绩）

{
  "v2": {
    "rules": 119,
    "joint": 99,
    "label": "post-hoc representation-only diagnostic, not primary"
  },
  "v4_clean": {
    "rules": 93,
    "joint": 93,
    "label": "post-hoc representation-only diagnostic, not primary"
  },
  "v4": {
    "rules": 108,
    "joint": 108,
    "label": "post-hoc representation-only diagnostic, not primary"
  }
}

## 解释边界

共同通过使用冻结的窄解释检查，不等于全面事实真实性。仍需人工复核时长、天数、数量、室内事实与可行题误拒绝。
本轮仅比较两组SFT候选与v2；未新增付费API基线、未训练DPO。数据清洗含隔离、提示词迁移与解释改写，不能把相对v2的变化归因于单一因素。
