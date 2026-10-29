# 景点属性对最终路线的影响：探索性真实模型对照

同一候选池、天气、模型、生产联合 Planner→道路核验→审核/返工→时间核查→最终输出，仅切换属性开关。道路使用已保存的真实响应，RAG 关闭，贴士分支两组固定为空，不计入模型质量。
10 条作者预设请求，南京/上海/三亚，包含兴趣、雨天、老人避台阶偏好及两条证据不足/冲突拒绝对照。标签为作者草稿，未人工确认；不是正式冻结测试，不给泛化显著性结论。评分读取最终时间线，不以说明文字变长作为改善。每例仅安排一个景点，不能用本实验证明多景点动线或通勤质量提升。
两组评分使用同一份独立参考属性，不能因实验组多了 indoor=true 就直接改分；硬规则检查部分复用生产代码，不能视为独立人工质量判定。室内标签只适用于有证据的展厅参观，不能保证排队、庭院或通勤避雨。
所有真实调用及失败保留账本，费用 unpriced；前26次调用的底层HTTP重试次数不可统计，随后修复实际SDK客户端关闭重试。Token 仅统计响应实报用量，缺失不会补零。重复运行非独立新请求。切换重试保护及并发时的中断保留，不能算作属性收益。

| 指标 | 无属性 | 有属性 |
|---|---:|---:|
| 正常请求任务成功（含中断） | 6/8 (75.00%) | 8/8 (100.00%) |
| 含拒绝对照的预期结果正确（含中断） | 7/10 (70.00%) | 10/10 (100.00%) |
| 负例明确安全拒绝 | 1/2 (50.00%) | 2/2 (100.00%) |
| 不合格路线放行 | 0/10 (0.00%) | 0/10 (0.00%) |
| 执行失败 | 2/10 (20.00%) | 0/10 (0.00%) |
| 其中实验工具中断 | 2/10 (20.00%) | 0/10 (0.00%) |

记录配对 10 组，其中两组均无执行错误、可比较质量的配对 8 组；改善 1 组，退化 0 组。

| 请求 | 重复 | 组别 | 最终选点 | 预期结果正确 | 返工 | 问题 |
|---|---:|---|---|---|---:|---|
| nanjing-rain | 0 | without_attributes |  | False | 3 | empty_route, interest_mismatch, route_structure  |
| nanjing-rain | 0 | with_attributes | 南京博物院 | True | 0 |   |
| nanjing-imperial-exam | 0 | without_attributes | 中国科举博物馆(江南贡院) | True | 0 |   |
| nanjing-imperial-exam | 0 | with_attributes | 中国科举博物馆(江南贡院) | True | 0 |   |
| nanjing-lake | 0 | without_attributes | 玄武湖景区 | True | 0 |   |
| nanjing-lake | 0 | with_attributes | 玄武湖景区 | True | 0 |   |
| shanghai-rain | 0 | without_attributes |  | False | 0 | empty_route, interest_mismatch HarnessInterrupted |
| shanghai-rain | 0 | with_attributes | 上海自然博物馆 | True | 0 |   |
| shanghai-evolution | 0 | without_attributes | 上海自然博物馆 | True | 0 |   |
| shanghai-evolution | 0 | with_attributes | 上海自然博物馆 | True | 0 |   |
| shanghai-garden-rain-conflict | 0 | without_attributes |  | True | 3 | empty_route, required_poi_missing, route_structure  |
| shanghai-garden-rain-conflict | 0 | with_attributes |  | True | 3 | empty_route, required_poi_missing, route_structure  |
| shanghai-elder | 0 | without_attributes | 上海海洋水族馆 | True | 0 |   |
| shanghai-elder | 0 | with_attributes | 上海自然博物馆 | True | 0 |   |
| sanya-rain-evidence-shortage | 0 | without_attributes |  | False | 0 | candidate_pool_preference_conflict, empty_route HarnessInterrupted |
| sanya-village | 0 | without_attributes | 三亚西岛海洋文化旅游区 | True | 0 |   |
| sanya-village | 0 | with_attributes | 三亚西岛海洋文化旅游区 | True | 0 |   |
| sanya-rain-evidence-shortage | 0 | with_attributes |  | True | 1 | candidate_pool_preference_conflict, empty_route, route_structure  |
| sanya-coastal-rocks | 0 | without_attributes | 天涯海角游览区 | True | 0 |   |
| sanya-coastal-rocks | 0 | with_attributes | 天涯海角游览区 | True | 0 |   |

## 用量、耗时与指纹

```json
{
  "protocol": "poi-attributes-route-pilot-v1",
  "status": "small_pilot_closed_at_user_request",
  "model": "gpt-6.1-sol",
  "max_calls": 240,
  "attempted_calls": 98,
  "summary": {
    "arms": {
      "without_attributes": {
        "task_success_non_rejection": {
          "numerator": 6,
          "denominator": 8,
          "percent": 75.0
        },
        "expected_outcome": {
          "numerator": 7,
          "denominator": 10,
          "percent": 70.0
        },
        "safe_rejection_negative": {
          "numerator": 1,
          "denominator": 2,
          "percent": 50.0
        },
        "unsafe_release": {
          "numerator": 0,
          "denominator": 10,
          "percent": 0.0
        },
        "execution_failures": {
          "numerator": 2,
          "denominator": 10,
          "percent": 20.0
        },
        "harness_interruptions": {
          "numerator": 2,
          "denominator": 10,
          "percent": 20.0
        },
        "mean_rework": 0.6,
        "mean_elapsed_ms": 43010.71084002033,
        "measured_usage_trials": 8,
        "actual_input_tokens_known_only": 266705,
        "actual_output_tokens_known_only": 9932,
        "model_calls": 42
      },
      "with_attributes": {
        "task_success_non_rejection": {
          "numerator": 8,
          "denominator": 8,
          "percent": 100.0
        },
        "expected_outcome": {
          "numerator": 10,
          "denominator": 10,
          "percent": 100.0
        },
        "safe_rejection_negative": {
          "numerator": 2,
          "denominator": 2,
          "percent": 100.0
        },
        "unsafe_release": {
          "numerator": 0,
          "denominator": 10,
          "percent": 0.0
        },
        "execution_failures": {
          "numerator": 0,
          "denominator": 10,
          "percent": 0.0
        },
        "harness_interruptions": {
          "numerator": 0,
          "denominator": 10,
          "percent": 0.0
        },
        "mean_rework": 0.4,
        "mean_elapsed_ms": 48250.11327995453,
        "measured_usage_trials": 10,
        "actual_input_tokens_known_only": 325670,
        "actual_output_tokens_known_only": 10992,
        "model_calls": 42
      }
    },
    "paired_trials": 10,
    "quality_comparable_pairs": 8,
    "improved_pairs": 1,
    "regressed_pairs": 0,
    "changes": [
      {
        "case_id": "nanjing-rain",
        "repetition": 0,
        "before": [],
        "after": [
          "南京博物院"
        ],
        "before_correct": false,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": true
      },
      {
        "case_id": "nanjing-imperial-exam",
        "repetition": 0,
        "before": [
          "中国科举博物馆(江南贡院)"
        ],
        "after": [
          "中国科举博物馆(江南贡院)"
        ],
        "before_correct": true,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": false
      },
      {
        "case_id": "nanjing-lake",
        "repetition": 0,
        "before": [
          "玄武湖景区"
        ],
        "after": [
          "玄武湖景区"
        ],
        "before_correct": true,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": false
      },
      {
        "case_id": "shanghai-rain",
        "repetition": 0,
        "before": [],
        "after": [
          "上海自然博物馆"
        ],
        "before_correct": false,
        "after_correct": true,
        "quality_comparable": false,
        "selection_changed": true
      },
      {
        "case_id": "shanghai-evolution",
        "repetition": 0,
        "before": [
          "上海自然博物馆"
        ],
        "after": [
          "上海自然博物馆"
        ],
        "before_correct": true,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": false
      },
      {
        "case_id": "shanghai-garden-rain-conflict",
        "repetition": 0,
        "before": [],
        "after": [],
        "before_correct": true,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": false
      },
      {
        "case_id": "shanghai-elder",
        "repetition": 0,
        "before": [
          "上海海洋水族馆"
        ],
        "after": [
          "上海自然博物馆"
        ],
        "before_correct": true,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": true
      },
      {
        "case_id": "sanya-rain-evidence-shortage",
        "repetition": 0,
        "before": [],
        "after": [],
        "before_correct": false,
        "after_correct": true,
        "quality_comparable": false,
        "selection_changed": false
      },
      {
        "case_id": "sanya-village",
        "repetition": 0,
        "before": [
          "三亚西岛海洋文化旅游区"
        ],
        "after": [
          "三亚西岛海洋文化旅游区"
        ],
        "before_correct": true,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": false
      },
      {
        "case_id": "sanya-coastal-rocks",
        "repetition": 0,
        "before": [
          "天涯海角游览区"
        ],
        "after": [
          "天涯海角游览区"
        ],
        "before_correct": true,
        "after_correct": true,
        "quality_comparable": true,
        "selection_changed": false
      }
    ]
  },
  "fingerprints": {
    "cases": "4e252c37f1c073a08809ff2e261868b8a192f29102e6392863d214ad3e5a49e1",
    "maps": "46d290bca3ba74fa3879e62b9112e4d93d2eeb8e4770d82bc66e7b5c5e2ccb12",
    "attributes": "1cb3e106058b3bde7b7c4565af4432fe0ed81e8ceeddca39e0e096375ac77c2e",
    "code": {
      "app/planning/nodes.py": "f3c9b0c29989e158f5cadbfb8d225f6fa26e4ee7efba561ac1ac85f4f68f1bd1",
      "app/planning/poi_attributes.py": "d5b844d0e83cb1dd9cccc709c27309dc9f793111e260969c3d2d4e5bb9725389",
      "app/planning/helpers.py": "e02be852f963edb7329697d0f7e8481a749e2225ed146b9b912f85046dc303bc",
      "app/planning/constraints.py": "d0a24fa46f3a7c5c1d1323a7fd13715b1e005fe11e678b3718a18ba01371cca2",
      "app/evaluation/route_interest.py": "738e03d938a28926f280c0f9b73ab1653a336472a709e73a6212823f26ef4f1e",
      "scripts/evaluate_poi_attributes.py": "04e2be97f35b702688b408b6aab40d916345549d5fa043613363d20418ea446b"
    }
  }
}
```

复现：`python scripts/evaluate_poi_attributes.py` 先校验并预览；`python scripts/evaluate_poi_attributes.py --run --repetitions 2 --max-calls 240 --out data/poi_attributes_v1/new_run` 运行真实模型。新目录防止覆盖账本。复现相同条件不保证模型逐字一致。
