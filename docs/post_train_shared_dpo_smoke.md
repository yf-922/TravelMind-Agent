# 三底座 DPO 冒烟测试

2026-10-26 15:30（北京时间）在 jiale 服务器 GPU0 启动。

- 启动器：`scripts/launch_shared_dpo_smoke.py`。
- 顺序：v2-current → A → B；任一组失败即停止，不自动执行全量训练。
- 共用冻结数据：346 对训练、54 对验证；冒烟各取最多 16 对，5 步，batch=1、梯度累积=8、cutoff=3584、4-bit QLoRA、sigmoid DPO。
- A、B 分别在 CPU 合并对应 SFT 权重后再创建新的 DPO LoRA，使禁用 DPO 适配器时的参考模型确实是各自的 SFT 模型。临时合并目录仅由本次任务创建，逐组释放；不删除历史模型。
- 数据文件和 A/B SFT 适配器 SHA256 在启动前校验。
- 相关本地测试：6 passed（shared DPO 与低显存 DPO 数值实现）。

## 停止人工监视时的快照

15:32:43，v2 已完成 3/5 步，约 28 秒/步；loss 为 0.6931、0.6724、0.6956，均有限。GPU0 当前使用约 21,131 MiB，利用率 100%，尚无 OOM。此为短流程稳定性检查，不能据此判断 DPO 效果提升，也尚不能宣称 A/B 已通过。

按用户要求此时停止人工轮询，后台继续三组冒烟。估计三组在 15:42–15:52 完成，包含 A/B CPU 合并及验证；实际以持久化结果为准。

服务器结果路径（根目录 `/home/jiale/TravelMind-posttrain`）：

- `logs/shared-dpo-smoke-20261026-status.json`：当前阶段/失败原因。
- `logs/shared-dpo-smoke-20261026-results.json`：各组完成后的 loss、评估、PyTorch 显存峰值及耗时。
- `logs/shared-dpo-smoke-20261026-{v2,A,B}.log`：训练日志。
- `logs/shared-dpo-smoke-20261026-restore.json`：结束时恢复原 Base vLLM 的启动记录；启动记录本身不代表健康检查通过。

后续先核查三组均完成 5 步、评估有限且无 OOM，再决定全量 DPO。冒烟产物不作为正式模型，也不用于效果结论。

## 完成结果

三组于北京时间 15:41:46 全部完成，总耗时 667 秒（11 分 7 秒）。每组均完成 5 步、保存适配器、loss 有限且无 OOM。

| SFT 底座 | 训练进程耗时（含加载/验证） | 平均训练 loss | 验证 loss（16对） | PyTorch 分配/保留显存峰值 GiB |
|---|---:|---:|---:|---:|
| v2-current | 184秒 | 0.6856 | 0.6956 | 19.43 / 20.29 |
| A | 185秒 | 0.6954 | 0.6783 | 19.43 / 20.29 |
| B | 186秒 | 0.6960 | 0.6971 | 19.43 / 20.29 |

原 Base vLLM 已恢复，`/health` 和 `/v1/models` 可访问，模型为 `Qwen2.5-7B-Instruct`。临时合并权重已释放；全量 DPO 尚未启动。机器显存占用高来自恢复的 vLLM。

完整结果同步至 `data/post_train/shared_dpo_20261026/smoke_results.json`。16 对、5 步不用于比较三底座效果；全量训练前仍应覆盖最长样本的显存边界，再按共同数据/参数训练，并在独立任务集比较 SFT 与 DPO。
