# 受控路线规划 Benchmark 协议

执行器：scripts/benchmark_controlled.py。每个模型使用同一份冻结 50 场景、seed 42/43/44，共 150 条；先取 5 条 warm-up，warm-up 不进入统计。正式评测使用 temperature=0、固定 max_tokens=2048。延迟实验必须 concurrency=1；吞吐实验另行固定并发度，不与延迟结果混合。

服务器启动 vLLM 时，所有版本必须采用同一权重形态：全部加载合并后的 BF16，或全部作为同 rank LoRA 挂在同一 Base 上。启动参数必须包含 --no-enable-prefix-caching。每次切换模型后等待 /v1/models 就绪，再进行 warm-up。SFT 与 DPO 不能用 merged-vs-LoRA 直接比较延迟。

脚本为每次请求记录 prompt/completion tokens、输出字符数、TTFT、E2E、TPOT、seed、模型、轮次、原始 JSON 和规则故障。summary.json 输出 Delivery、All-Rule、候选池忠实度、开放时间、时间轴/通勤、无重复、返工触发/轮次/恢复率、token 平均值、P50/P95、TPOT P50。

## 结果解读模板

“在相同权重形态、相同 vLLM 参数、关闭 prefix cache、固定 5 条 warm-up、温度 0、同一 50×3 输入下，模型 A 的 All-Rule Pass Rate 从 X% 提升到 Y%，其中 Opening Hours 提升 Z 个百分点，Timeline & Transit 提升 W 个百分点。平均 Completion Tokens 为 … 与对照 … 基本一致，排除输出长度膨胀；Rules+Rework 的总 Token 从 … 降至 …，返工恢复率为 …。E2E P50/P95 和 TPOT P50 分别为 …，因此质量收益与推理效率应分开报告。若 paired bootstrap 区间跨 0，则只能称为方向性改善，不能称为显著提升。”

当前 50 题没有酒店、航班或工具返回值，不能从它计算开放域幻觉率；当前可计算的 Hallucination 仅限候选池外实体率。
