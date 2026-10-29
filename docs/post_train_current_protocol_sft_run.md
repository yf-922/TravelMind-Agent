# 当前生产协议 v1/v2 SFT 重训（2026-10-25）

本次从原始 `Qwen2.5-7B-Instruct` 分别训练两版 LoRA，使用新教师答案及当前生产消息，旧模型不覆盖。只执行 SFT，不自动执行 DPO 或恢复已停止的测试。

## 数据和训练设置

| 项目 | v1-current | v2-current |
| --- | ---: | ---: |
| 训练样本 | 156 | 250 |
| 共用独立验证样本 | 30 | 30 |
| 导出序列最长 Token | 2411 | 3008 |
| 训练轮数 | 3 | 3 |
| 预期优化步数 | 30 | 48 |

两版各自从同一原始基座开始；验证集不进入训练，最终五城市测试集不改动。保持历史超参数：LoRA rank 16 / alpha 32 / all，BF16，SDPA，gradient checkpointing，batch 1 × 累积16，学习率 1e-4，cutoff 3584，seed 42。按照 epoch 验证和保存，选择最低验证 loss 的 checkpoint，CPU 合并。相同 epoch 意味着不同步数；效果比较时需一起报告样本量、实际监督 Token、步数，不能把差异全部归因为数据内容。

## 冒烟及启动结果

2026-10-25 15:32 启动 GPU0 队列。检查上传文件 SHA256、导出长度、两版验证集一致性后，分别选取最长的16条训练样本和4条验证样本进行5步冒烟（累积1）。

| 冒烟 | 步数 | 验证 loss | CUDA 分配峰值 GiB | CUDA 预留峰值 GiB | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | 5 | 1.079459 | 19.256 | 20.207 | 通过 |
| v2 | 5 | 1.094143 | 20.370 | 21.518 | 通过 |

显存峰值来自 PyTorch，并非 `nvidia-smi` 的整卡总占用。冒烟验证的是工程可运行性，不是最终规划质量。

两版冒烟均通过后，15:33 自动启动 v1 全量训练，随后顺序执行 v1 CPU 合并、v2 全量训练、v2 CPU 合并。15:35 已观察到 v1 完成6/30步，loss有限，约16秒/优化步，无OOM。初估全流程北京时间16:00–16:10完成，未包含路线效果评测。第一轮全量验证和保存通过后停止人工轮询。

15:37 稳定性检查通过：v1已完成11/30步，第一轮30条完整验证的loss为0.836971，`checkpoint-10`成功保存，继续训练正常。此时停止人工轮询，后台顺序队列继续运行；预计两版训练与合并在16:00–16:10完成。

## 完成核查（16:02）

15:59 两版训练与CPU合并全部成功，整个队列耗时1566.49秒（约26.1分钟）。原始Base vLLM服务已恢复，`/v1/models`已返回正常模型列表。尚未执行新版本的路线效果评测，也未执行DPO。

| 指标 | v1-current | v2-current |
| --- | ---: | ---: |
| 完成优化步数 | 30 | 48 |
| epoch 1 验证loss | 0.836971 | 0.718484 |
| epoch 2 验证loss | 0.689081 | 0.595130 |
| epoch 3 验证loss | 0.648724 | 0.567109 |
| 选中checkpoint | 30 | 48 |
| 训练进程耗时（秒） | 523.58 | 895.59 |
| CUDA分配峰值（GiB） | 19.407 | 20.523 |
| CUDA预留峰值（GiB） | 20.363 | 21.713 |
| 实际样本呈现次数 | 468 | 750 |
| 实际非padding输入输出Token | 971871 | 1703193 |
| 实际参与loss的监督Token | 178329 | 281259 |

两版均在相同30条验证集上逐epoch下降，本轮没有出现验证loss反弹；这不足以排除过拟合，也不能推出路线约束指标的排名。v2验证loss更低，但其样本量、监督Token与优化步数均更多，不能把差异单独解释为数据质量提升。

下一步应以原始Base、v1-current、v2-current及v2-large四组，在冻结五城市50题×3 seeds上使用当前生产消息、相同解码和结构化约束，分别比较Planner-Only与Rules+Rework。每模型每模式150条，共1200条最终评测记录，返工调用另计。历史API指标仅在请求协议与评分一致时纳入同表，不能直接拼接异协议分数。由于这套50题已反复用于诊断，应标为开发基准；最终泛化结论还需新的未参与调参的独立测试。

## 运行位置和复核

- 服务器根目录：`/home/jiale/TravelMind-posttrain`；仅 GPU0。
- 启动器：`scripts/launch_current_protocol_sft.py`。
- 数据：`data/post_train/release_current_protocol_v1_v2/{v1,v2}`。
- 配置：`configs/post_train/qwen25_7b_planner_sft_{v1,v2}_current.yaml`。
- 状态：`logs/current-20261025-status.json`。
- 启动日志：`logs/current-20261025-launcher.log`。
- 训练日志：`logs/current-20261025-{v1,v2}-full.log`。
- 实际训练 Token：`logs/current-20261025-{v1,v2}-full-tokens.jsonl`。
- 训练进度：`saves/TravelMind-Planner-{v1,v2}-current-20261025/trainer_log.jsonl`。
- LoRA：`saves/TravelMind-Planner-{v1,v2}-current-20261025`。
- 合并模型：`models/TravelMind-Planner-{v1,v2}-current-20261025`。

所有子阶段失败即停止后续训练并写入失败状态；完成或失败后恢复原始 Base vLLM 服务。日志与训练进程在 SSH 断连后保留。禁止重复启动同名队列以免覆盖产物。

训练结束后，应先核查 best checkpoint、验证 loss、实际 Token、显存和合并产物，再使用冻结的五城市50题×3 seeds统一协议，比较 Planner-Only 和 Rules+Rework 的规划质量；验证 loss不能替代硬约束、软约束、幻觉和返工率等指标。
