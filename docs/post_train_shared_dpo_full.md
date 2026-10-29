# 三底座全量 DPO

用户授权后于北京时间 2026-10-26 15:46 启动，GPU0，顺序 v2-current → A → B。

启动命令：`python scripts/launch_shared_dpo_smoke.py --full`。三组均独立从各自 SFT 开始，不延续五步冒烟适配器。

共同配置：冻结训练 346 对、验证 54 对；2 epoch、88 优化步；4-bit QLoRA、rank16/alpha32、target all、batch1、梯度累积8、cutoff3584、学习率5e-6、beta0.1、sigmoid、seed42。每 epoch 验证和保存，保留最多2个检查点，最终加载验证 loss 最低的检查点。

A、B 的 SFT 权重临时在 CPU 合并；新建 DPO 适配器，禁用该适配器时参考模型为对应 SFT。合并目录逐组释放，历史模型不删除。正式 DPO 适配器如引用已释放临时目录，后续部署须依据 `binding.json` 重建对应 SFT 合并底座；不能直接挂在原始 Qwen 上。

15:48 左右停止人工监视时：v2 已完成3/88步，每步约29秒，loss依次0.6931、0.6844、0.6642，当前GPU显存21791 MiB、利用率100%，未见OOM或非有限loss。仅确认起步稳定，尚未覆盖完整数据，不能保证后续所有长度均不OOM。

预计每组45–50分钟（含54对验证、保存和加载），三组共约2小时15分–2小时30分；预计北京时间18:00–18:20结束，依实际日志更新。失败则停止后续组并记录错误；结束恢复原Base服务，不自动进行效果评测。

服务器根目录：`/home/jiale/TravelMind-posttrain`。

- 状态：`logs/shared-dpo-full-20261026-status.json`
- 分组日志：`logs/shared-dpo-full-20261026-{v2,A,B}.log`
- 输入/底座绑定：`logs/shared-dpo-full-20261026-{v2,A,B}-binding.json`
- 结果：`logs/shared-dpo-full-20261026-results.json`
- 产物：`saves/shared-dpo-full-20261026-{v2,A,B}`

完成后先核查三组88步、验证及权重完整性，再在同一冻结独立集上对照各自SFT：Planner-Only和Rules+Rework，报告非法时间率、可行/无解判别、约束通过率、返工次数、响应时延与Token成本。训练loss不等同于路线质量。
