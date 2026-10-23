# Reviewer、门控与局部重规划：证据化评测 v3

## 1. 不预设结果的软约束挑战

原始 `reviewer_ablation_real_v2.json` 不修改。新版本只运行 6 个受控草案，
不是将人工注入错误冒充自然 Planner 失败：

- 4 个可满足语义缺陷：老人登阶、38 度正午暴晒、隐式排斥商业主题、跨区折返。
- 1 个不可满足对照：仅剩登阶活动候选，但老人不能登阶。
- 1 个正常对照：玄武湖单点轻松游，用于观察误拒绝。

离线预检必须先证明草案通过生产硬约束而不符合预设语义标签。反感商业化
不是认定某景点普遍不好；老人用例限定本次活动为登阶游览，不假定整个景区
没有无障碍设施。折返按保存的实际道路距离矩阵计算：当前顺序距离超过
相同景点最短排列 30% 记为不必要折返，不使用直线距离替代道路。
高温用例预设 11:00–14:00 不安排露天游览，是用例标签，不是全国统一规则。

三组共用草案、模型和地图回放。仅 Reviewer 组读取语义审核意见；评分标签
不进入 Planner/Reviewer Prompt。最多三次返工，无隐藏调用重试。每次记录
生产输入 Prompt、实际 Provider usage、结构化输出、各轮状态和门控决定。

组合表复用 25 条未改变的历史结果，加 5 条受控挑战；正常对照单列。
必须标记为“历史自然请求＋受控挑战组合”，不能改称原 30 条自然请求成功率。
不可满足用例正确拒绝计为安全行为，不计为生成了成功行程。

所有新标签保留 `pending_human_review`。用户逐项核对后，通过绑定用例
指纹的导入流程生成单独的 verified 副本；不会将旧标签批量改为人工确认。

```powershell
python scripts/evaluate_soft_review.py
python scripts/evaluate_soft_review.py --execute --max-calls 42
python scripts/review_soft_review_labels.py export
# 人工编辑 data/eval/soft_review_v3_human_review.json，填写身份、带时区时间和逐项确认
python scripts/review_soft_review_labels.py import
```

## 2. 当前门控：真实 usage 与离线路由分开

从原始 30 条 task 读取真实草案，回放道路核验，再运行当前门控，不关闭
语义审核。硬约束通过不等于无语义风险；偏好会触发 `preference_review`。

Reviewer 的实际调用和输入/输出 Token 取自账本 usage。历史实验未调用
Time Check、未执行自适应对照、未保存所有修复中间状态，因此这些缺项
保持不可统计。Tokenizer 不能补造未执行的模型输出。当前初稿全部需要
Reviewer 时，如实报告初稿可跳过次数和 Token 为零。

```powershell
python scripts/audit_current_risk_gate.py
```

结果覆盖 `evaluation/adaptive_routing_replay.json/.md`；旧策略历史脚本仅用于
重现旧结果，不用于当前简历节省指标。

## 3. 真实状态与生产 Prompt，而非成本常量

从同一账本选取 30 个不同的普通任务初稿，保留解析需求、天气、候选与路线。
构造晚出发/下午室内的局部修改，不能复制一份状态凑样本数。

- 完整路径：生产 Intent Prompt、Query Rewrite Prompt、回放保存的解析字段
  与天气/POI JSON、生产联合 Planner Prompt；不携带旧路线作为初稿。
- 局部路径：与修改 API 共用 `restore_modification_state`，保留原路线供修改，
  使用真实生产 Planner Prompt，不再构造前置 Prompt。

按 `tiktoken==0.14.0 / cl100k_base` 对实际 system/human 文本分词，合并各节点
文本数后计算 `1 - local/full`。不统计工具 Schema、消息协议边界或输出，
没有声称该 tokenizer 等于 Grok 的计费 tokenizer。Prompt 完整副本保留在
被忽略的 `data/eval/checkpoint_prompt_capture.json`，报告绑定其哈希。

每状态本地计时重复三次，测序列化、状态恢复、Prompt 构造，不加 sleep。
没有天气/POI历史耗时 Trace，因此不使用示例 150ms/100ms/800ms 填值。
每次省去两个前置 Prompt 调用机会及天气、POI两类查询；不能据此说一定
省两次 HTTP，因为真实检索可能分页、重试或命中缓存。

```powershell
python scripts/offline_engineering_benchmark.py --runs 30 --checkpoint-samples 30 --checkpoint-repetitions 3
python -m pytest -q tests/test_evidence_benchmarks.py tests/test_offline_engineering_benchmark.py
```

道路并发基准仍单独采用固定延迟服务来测调度，不能与上述本地计时合成
端到端收益；原先的阶段模拟 31.8% 和固定预算 15.2% 已撤销。
