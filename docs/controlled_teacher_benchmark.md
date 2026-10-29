# 教师 API 同协议评测

入口：`python -u scripts/benchmark_teacher_controlled.py --modes planner-only`。按最新用户要求，仅执行教师 Planner-Only（150条），不执行教师 Rules+Rework。程序默认也是 planner-only；只有显式指定才会执行其他模式。从 `.env.local` 读取模型、Base URL 和密钥，不在命令行、账本或报告保存密钥。不使用 GPU，不修改正在运行的本地模型评测。

## 固定的对比条件

- 与 `benchmark_controlled.py` 使用同一冻结 cases、messages、道路回放、TravelRoute Schema、`score_case` 和返工反馈文字。
- 50 场景，各用 42/43/44 三个 seed，temperature=0。它们是重复测量，不是 150 个独立场景。
- Planner-Only 与 Rules+Rework 各 150 条，后者最多额外返工 3 轮。
- 每个模式 5 次预热，不计正式指标；并发 1，输出上限 2048，流式 SSE，180 秒网络超时。
- 为保持与当前本地实验行为一致，Schema 解析失败结束该样本，不额外返工；规则失败才反馈重排。
- 本次单独计数、包含预热和中断请求在内最多 760 次 API 尝试。仅明确的服务暂不可用或 HTTP 429/502/503/504 最多额外重试三次，每次等待 60 秒。其他错误立即停止，不静默换接口或改参数。退避计入正式样本端到端时间，所有尝试均记录。

## 不能视为已控制的条件

API 的底层硬件、负载、调度、缓存及内部提示词无法控制。请求 reasoning_effort=none，但代理返回的 usage 仍可能有 reasoning tokens，须明确报告。API 的 P50/P95 是调用方端到端延迟；不能将其与服务器单卡的差异直接归因于模型权重。

Provider tokenizer 与 Qwen tokenizer 不同，因此跨模型 Token 数直接用于各自调用消耗统计，不足以独立证明回答更长或更短；同时报告输出字符数。reported completion tokens 含 reasoning 时另报 reasoning 与 visible completion。单个正文分块不足以测量逐 Token 生成过程，TPOT 置空；其他多分块 TPOT 仍是代理可观察值，不是 GPU 内核速度。

## 落盘和恢复

输出目录：`data/post_train/controlled_teacher_20261027/`。

- `binding.json`：模型及协议绑定，冻结输入和 Schema 指纹。
- `teacher.calls.jsonl`：每次调用开始/完成/失败记录、原始请求消息、正文、实际 usage、流式分块数及返回模型名；含预热。
- `teacher_{mode}/results.jsonl`：逐样本每一轮输出、确定性错误及计时；不含预热。
- `teacher_{mode}/summary.json`：每种模式完成后生成。
- `status.json`：逐样本进度；遇到传输错误记录 stopped_on_error。
- `summary.json`、`report.md`：两种模式完成后生成。
- `api_accounting.json`：包含预热、失败及中断次数的 API 总消耗；无法获知失败调用实际计费，明确标记未知。

相同命令可以跳过已完成样本恢复；若进程被外部杀死，先确认旧进程确实退出，再核对未完成调用（其计费未知），最后清理该目录的 `run.lock`。不要在旧进程仍运行时清锁或重复启动。

## 指标审计

综合通过率使用相同的完整 rubric，同时给出全样本分母与已交付分母两种口径。未交付样本不得获得子规则通过分；跨日重复也计入重复错误。缺失 usage 不猜测 Token 数，不可测 TPOT 不填零。

当前本地运行器的历史子项汇总存在未交付样本被标为通过的问题，最终横向对比必须从各模型原始 rounds/faults 重新统一汇总，不能直接拼接旧 summary。返工触发率按实际发出第二轮请求统计；Schema 失败没有重试，所以不必等于 1−首轮通过率。

本轮依然是多次使用过的五城市开发基准，不能称为未参与调参的最终独立测试。
