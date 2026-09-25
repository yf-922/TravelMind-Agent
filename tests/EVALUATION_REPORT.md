# TravelMind 多 Agent 评估报告（v1）

## 评测范围

- 三层测试：单元、集成、端到端。
- 自动用例：10 条端到端用例，覆盖北京/重庆、文化游、慢节奏、雨天、亲子、预算有限等正常与边界表达。
- 运行方式：`python scripts/generate_eval_report.py`。
- 可复现数据：`evaluation/latest_report.json`。本次使用 Fixture POI 和离线规则 Agent，因此不受外部 LLM、地图网络和 Key 配额影响。

## 单元测试（3 个 Agent）

| Agent            | 断言重点         | 边界/异常      |
| ---------------- | ------------ | ---------- |
| IntentAgent      | 目的地字段、独立私有记忆 | 空历史记忆初始化   |
| POIResearchAgent | 仅返回工具提供的候选项  | 工具候选字段完整性  |
| PlannerAgent     | 行程景点属于候选池    | 禁止编造候选池外地点 |

## 端到端 Golden Set（10 条）

| ID     | 用户输入     | 期望输出 | 通过标准                           |
| ------ | -------- | ---- | ------------------------------ |
| E2E-01 | 北京文化一日游  | 完整行程 | 四个 Agent 均被调度，行程非空，Reviewer 通过 |
| E2E-02 | 重庆慢节奏+小吃 | 完整行程 | 同上                             |
| E2E-03 | 北京亲子少走路  | 完整行程 | 同上                             |
| E2E-04 | 重庆雨天游    | 完整行程 | 同上                             |
| E2E-05 | 北京经典景点   | 完整行程 | 同上                             |
| E2E-06 | 重庆半日游不紧凑 | 完整行程 | 同上                             |
| E2E-07 | 北京预算有限   | 完整行程 | 同上                             |
| E2E-08 | 重庆朋友同行   | 完整行程 | 同上                             |
| E2E-09 | 北京历史建筑   | 完整行程 | 同上                             |
| E2E-10 | 重庆首次旅行   | 完整行程 | 同上                             |

## 本次结果与局限

JSON 文件记录成功率、每条延迟、调度链路与失败分类。该基线验证的是多 Agent 编排正确性，不等同于真实 LLM 的旅游建议质量；真实线上评测应额外记录高德/票价/天气工具的错误率，以及由人工或 LLM-as-Judge 给出的约束满足评分。

## LLM-as-Judge（已实现）

- 脚本：`python scripts/run_llm_judge.py --allow-external-calls`；小额试运行：`python scripts/run_llm_judge.py --limit 1 --allow-external-calls`。
- 固定项：Golden Set、Prompt 版本 `travelmind-judge-v1`、temperature=0、Pydantic 评分 Schema。
- 评分维度：候选池事实一致性、需求满足、行程完整性、Reviewer 一致性、表达清晰度，各 1-5 分。
- 产物：`evaluation/llm_judge_report.json`，保存模型提供方、模型覆盖项、每条评分、理由、失败分类与 Judge 延迟。
- 注意：它是辅助评审，不替代人工抽查；应抽查至少 3 条评分理由是否引用了真实输入。

### 首轮真实运行结果（2026-07-27）

- Judge：DeepSeek（项目当前默认配置），10 条均成功得到结构化评分，平均 Judge 延迟约 3.00 秒。
- Judge 通过率：80%（8/10）。
- 失败分类：`constraint_miss` 2 条，均为“慢节奏/餐饮偏好”等自然语言约束在离线 Fixture 行程中只得到 3 分的基本满足判定。
- 下一步：把偏好字段显式传给 Planner；为餐饮、慢节奏、亲子、雨天等约束分别增加可自动断言的字段；抽查这 2 条失败理由并与人工评分对照。
