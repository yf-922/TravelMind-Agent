# 不可满足与信息不足场景的安全契约测试

新增独立资产 `tests/eval/refusal_cases.json`，预先定义期望风险项；不修改原 39 条回归集，也不把原有失败重新标为成功。

|场景|预期处理|本次结果|
|---|---|---|
|候选池为空|不批准空路线|拦截|
|全室内要求，但固定候选只有户外公园|不批准违反室内限制的路线|拦截|
|指定 10:00–11:00 游玩，唯一展馆 17:00 开放|不批准开放时间冲突|拦截|
|限制步行 1 公里，但道路核验缺失|标记无法核验，不批准已满足约束|拦截|

测试故意注入 Reviewer 批准、Time Check 无冲突的错误模型结果，执行生产风险门控、Reviewer、Time Check 和 Finalize，检查最终 approved=false、风险标记保留以及达到轮数上限后的退出分支。4/4 场景拦截，最终误批准 0/4；另有 1 条合法路线对照正常批准。

这是离线规则契约与故障注入，不是实际模型质量或完整重规划实验。固定候选无解不代表扩展搜索后仍无解；道路信息不足也不等于数学上无解。没有实际运行 Planner 或多轮图循环，因此这里只验证上限分支，不宣称验证完整收敛。

发现：部分失败路线的 `route_issues` 为空，风险原因仅保存在 `audit_policy.risk_flags` 中。当前不足以声称用户获得了清晰拒答、替代方案或完整降级解释。

可用于项目说明：新增 4 类不可满足/信息不足场景，验证审核模型错误批准时后端规则仍能阻止误批准。不能据此将旧结果改为 35/35 或笼统声称系统正确率 100%。

复现：

```powershell
python tests/eval/run_refusal_contract.py
python -m pytest tests/test_refusal_contract.py -q
python scripts/validate_eval_assets.py
python -m pytest -q
```

逐场景报告写入被忽略的 `data/eval/refusal_contract.json`。
