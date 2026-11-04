# B-prep：修改一致性实验准备

## 资产盘点

`data/eval/checkpoint_modifications_v2.json` 的 `checkpoint_replan.rows` 保存了 120 个探针的全量/局部 Prompt Token、候选池哈希、Mock 查询轨迹和本地构造耗时；`checkpoint_modifications_v2_prompts.json` 保存了 120 对 Prompt。账本没有保存两条路径实际调用 Planner 后的 JSON 输出，因此现阶段不能从 32.33% Prompt Token 缩减推断路线输出一致，也不重跑模型来补造历史输出。

## 预注册一致性口径

1. **字节级**：原始 JSON 文本完全相同。
2. **归一化 JSON**：递归排序对象键、归一空白后完全相同。
3. **行程语义**：逐日景点名称、开始时间和结束时间的顺序一致；这是主指标。

不一致分为：未修改部分漂移、修改部分响应差异、数量或顺序变化。数量和顺序变化单独归类，不把它误算成文本漂移。

运行 `python scripts/compare_checkpoint_outputs.py --dry-run` 会检查 120 个探针和 Prompt 对，并展示前 3 个探针的缺失输出状态；若未来获得两路 JSONL，传入 `--full-output` 与 `--local-output` 后可离线重算全部三级指标。
