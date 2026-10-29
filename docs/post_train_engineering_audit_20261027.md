# TravelMind-Agent 后训练工程审计报告

审计日期：2026-10-27。用途：算法工程面试、实验复盘和后续合理性评估。本文以真实配置、数据、运行代码、服务器训练产物和逐条评测记录为依据；规划文档中的目标不作为已经完成的结果。

## 证据范围与判读原则

- 本地仓库：`C:/Users/admin/Desktop/work/TravelMind-Agent-main`；远端：`/home/jiale/TravelMind-posttrain`。训练使用服务器 GPU0；本次审计没有启动训练、推理服务或付费 API。
- 关键原始证据已归档至 `data/post_train/engineering_audit_20261027/remote_snapshot.json`：六个正式任务的 `trainer_state.json`、实际 `training_args.bin` 中的配置、`adapter_config.json`、训练结果、远端实际 DPO YAML、安装版本及参考模型实现源码。JSON 路径 `runs.<版本>` 可定位。
- 当前统一比较来自 `data/post_train/representative_eval50_v1/sft_schema_control_20261027/comparison.json`，其 `source` 字段指向各组逐条 `results.jsonl`。DPO 和 SFT 运行时间不同，但此次汇总核对了共享协议。
- 证据优先级：实际训练状态/运行参数/逐条结果 > 执行器生成的配置与绑定记录 > 静态模板 > 计划和过程状态快照。未显式配置的项会标明默认值来源；找不到证据不作反推。
- 附件 `file_inventory.json` 为后训练相关代码、配置、文档和结构化证据的路径、字节数与 SHA256 清单；它是文件级索引，不表示每个历史产物都重新执行过。当前正式主线是 v2-current、A、B，不能将早期 v2、v3、large 的不同协议成绩混进同一横向表。

## 1. 基座模型与微调架构

|项目|核实结果|
|---|---|
|基座|`Qwen/Qwen2.5-7B-Instruct`；远端目录 `/home/jiale/models/Qwen2.5-7B-Instruct`；明确是 Instruct 版本|
|架构|`Qwen2ForCausalLM`；28 层、hidden size 3584、28 个 attention heads、4 个 KV heads、intermediate size 18944、vocab size 152064|
|参数量|日志为基座+适配器 7,655,986,688 参数，其中可训练 40,370,176（0.5273%）；相减得到基座 7,615,616,512，约 7.62B|
|SFT 精度与方式|未量化 BF16 基座 + LoRA；日志显示可训练适配器上转 FP32；没有做全量微调|
|DPO 精度与方式|bitsandbytes 4-bit QLoRA；BF16 计算；NF4、double quantization 为已安装 LLaMA-Factory 默认值|
|LoRA|三组 SFT/DPO 均为 r=16、alpha=32；dropout YAML 未配置，默认 0.0，实际 adapter_config 亦为 0.0；bias=none|
|挂载位置|`lora_target: all`，实际展开为 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`，并非只有 Q/V|
|训练框架|LLaMA-Factory 0.9.3 + Transformers/HF Trainer + TRL CustomDPOTrainer；DPO 有本项目低显存补丁，不是 Swift，也不是从零实现完整训练器|
|实际依赖|torch 2.9.0、transformers 4.52.4、peft 0.15.2、trl 0.9.6、accelerate 1.7.0、bitsandbytes 0.50.2、vLLM 0.11.2|
|Attention|正式 SFT/DPO 使用 `flash_attn: sdpa`；初始计划中的 FA2 不能当作正式运行配置|

参数选择的证据边界：配置体现单卡 24GB 下的小批量、梯度累积、checkpointing 和低秩适配策略；没有发现 rank、学习率或 beta 的系统搜索记录，不能宣称这些值是实验寻优所得。正式 DPO 记录显存 allocated 约 20.64 GiB、reserved 约 21.77 GiB，说明本次运行可装入 4090，不等于任意长度数据都不会 OOM。

## 2. 数据演进与构造逻辑

### 2.1 版本命名、样本数与训练路径

|版本|SFT 训练数据|验证数据|训练路径|关键区别|
|---|---|---|---|---|
|早期 v1|156 条|历史验证协议|原始 Qwen → v1|历史起点；不是本次六模型主比较的 v2-current|
|早期 v2|156 旧数据 + 96 新数据，清理 2 条后 250 条|30 条|原始 Qwen → 旧 v2|城市、天气、多日、开放窗口补强；其旧答案不可等同于当前协议答案|
|v2-current，本文简称 v2|250 条，重新由教师生成答案|30 条重新生成的答案|原始 Qwen → 48 步 SFT|输入采用当时当前生产消息；不继续沿用旧答案|
|A-SFT|96 条新增数量专项；这一阶段不回放 250 条旧数据|数量专项 24 条 + 旧任务 30 条|Qwen → v2 48 步 → A 48 步|累计 96 步；专项继续训练|
|B-SFT|250 + 96 = 346 条混合|同样 24 + 30 条|Qwen → B 96 步|累计 96 步；从原始底座混合学习|
|v2/A/B-DPO|每组都使用同一份 346 对偏好|同一份 54 对|各自对应 SFT → DPO 88 步|不是各自生成一套不同 DPO 训练集|

早期 v2 的数据审计给出南京70、三亚42、上海44、成都48、苏州46，共250条；两条新数据因解释时长不一致被隔离。这里的城市分布描述旧 v2 来源，不能代替后续数量专项分布。早期目标“800条”不是正式主线实际训练样本数。

`regen_v1_v2_current_protocol.py` 重新生成 250 条训练与 30 条验证答案，v1-current 使用其中 156 条训练子集；教师为 `gpt-6.1-sol`。教师不看到旧答案，清除旧路线、返工、审核状态，直接捕获生产 `messages`；兴趣知识增强关闭。只使用本地规则，不调用 LLM Reviewer；首轮失败最多一次规则修复，合格的最终答案配原始首轮输入，返工轨迹和批评文本不作为 SFT 目标。

此次重生成的调用账本记录 326 次启动、324 次完成、2 次失败；完成调用报告的合计 Token 为 1,577,270。`status.json` 中 1,567,472 是较早快照，不能替代调用账本。此数字只属于这次重生成，不代表整个项目 API 总开销，也不包含之后的本地推理 Token。

### 2.2 数量专项的正负可行性对照

`prepare_count_infeasible_ab.py` 构建 48 个训练配对族，每族一个不可行题、一个可行题，共 96 条；验证 12 族/24 条，另有诊断 20 族/40 条。

- 每题用真实已有 POI 子集，1–3 个候选、1–3 天、晴天 26°C、09:00–18:00、游览时长 60/75/90 分钟。
- 不可行题要求 K+1 个全程不重复景点，但只有 K 个候选；标准答案保留全部天数标签，各天景点为空，并明确数量缺口。
- 可行配对题要求 K 个，并给出符合开放时间等规则的确定性路线见证。标签由脚本证明与构造，新增教师调用为 0。
- 它主要补“数量不足识别与空解表达”，不能声称覆盖高温、未知室内属性等全部无解类型。
- 训练/验证来源族和消息做分离。但模型开发多次参考已有评测问题，不能将所有历史诊断集统称为从未参与调参的独立测试集。

### 2.3 清洗与训推协议

当前协议重生成的检查包括候选成员、道路耗时、开放窗口、高温/室内条件、数量、必去/排除、解释与路线引用和时长一致性、严格时间格式、无依据空解、无需求餐段、旧返工叙述，以及序列长度不超过 3584。作用是规则可核验，不是证明开放域语义完全正确。

SFT 数据带真实生产 system/human 上下文；训练输入有提示词是本项目实际设计。`train_on_prompt=False` 只是不对提示词 Token 求监督损失，并非把提示词从输入删除。捕获时消息一致与 API 外层约束一致应区分：当前生产适配器 `strict=True`，评测脚本 `strict=False`，不能声称请求载荷逐字节完全一致。

### 2.4 DPO chosen/rejected 的真实来源

主发布目录：`data/post_train/shared_dpo_20261026/release`。Chosen 来自 250 条当前协议教师答案 + 96 条确定性数量答案。Rejected 最初由合法答案做本地规则缺陷注入，再由 90 次小规模本地模型生成补充真实失败。

|来源|训练对数|占 346 对比例|补充说明|
|---|---:|---:|---|
|确定性合成负例|331|95.66%|在合法答案上注入一个可验证错误|
|真实模型失败替换|15|4.34%|v2 7 条、A 7 条、B 1 条；每条负例重新验证失败|
|验证偏好|54|单独验证集|全部为合成规则负例|

本地 pilot 是 30 个训练输入 × 三模型 = 90 次调用，temperature=0.3、seed=42，合计 177,233 个本地推理 Token、0 次新增教师调用。不是高温大规模采样，不是持续更新的在线 DPO，也不是人工偏好排序。pilot 的 22/30、22/30、29/30 只是训练题诊断，不能当作独立测试成绩。

|错误类别，合并 natural 前缀|训练对数|具体负例|
|---|---:|---|
|duplicate|36|重复景点，破坏跨日/全程去重|
|nonpositive_duration|37|结束时间等于开始时间|
|membership|36|替换为候选池外 POI|
|commute_deficit_5min|35|压缩相邻时段，使道路通勤差 5 分钟|
|opening_boundary_5min|37|越过开放窗口边界 5 分钟|
|time_format|33|例如去掉小时的前导零，JSON 语法仍合法但时间 Schema 不合法|
|missing_explicit_required_poi|36|遗漏显式必去景点|
|false_refusal|48|可行题错误拒绝规划|
|count_infeasible_partial|48|数量不可行却安排不满足要求的部分路线|

三模型共用冻结数据、chosen 过规则、rejected 不过规则、两侧长度均受 cutoff 检查；来源族训练/验证交叉为 0，已知测试消息精确重合为 0。局限：精确去重不等于语义隔离；没有专家主观偏好标注；最终发布集没有系统覆盖老人登阶等全部最初设想的缺陷。

注意发布 `audit.json` 的 `training_started:false` 是构建时静态字段；真实训练已完成，应以 trainer_state 为准。

## 3. SFT 参数、损失与收敛

|参数|v2-current|A 继续训练|B 混合训练|
|---|---|---|---|
|learning_rate|1e-4|1e-4|1e-4|
|num_train_epochs|max_steps=-1，显式 3|未配置，框架默认 3 被 max_steps 覆盖|同左|
|max_steps / 实际优化步|48|48，新阶段|96|
|实际 epoch|3.0|8.0|4.36994|
|每卡 batch × 累积 × world size|1 × 16 × 1|1 × 16 × 1|1 × 16 × 1|
|名义 Global Batch Size|16 条|16 条|16 条|
|cutoff_len|3584|3584|3584|
|lr_scheduler_type|未显式配置，实际默认 linear|同左|同左|
|warmup_ratio / warmup_steps|未配置，实际默认 0 / 0|同左|同左|
|optimizer / weight_decay / clip|默认 adamw_torch / 0 / max_grad_norm=1|同左|同左|
|seed / data_seed|42 / 42|42 / 42|42 / 42|
|BF16 / FP16 / checkpointing|true / false / true|同左|同左|
|验证与保存|每 epoch；最低 eval_loss 选最佳|每16步；load_best=false，最终权重|同 A|

Global Batch Size 是名义完整累积批大小；数据集尾批不足整批，不应以步数×16冒充精确样本曝光量。

Loss：`train_on_prompt` 未在 YAML 配置，已安装框架默认 False，只对 assistant completion 求损失；`mask_history` 默认 False，数据为单轮；`packing` 原始 dataclass 默认 None，但参数解析阶段对 SFT/DPO 解析为 False；`neat_packing=False`。不能仅凭 YAML 没写就声称已启用 packing。

|训练|首条记录 loss → 最后记录 loss|全程平均 train_loss|验证变化|实际训练时间|
|---|---|---:|---|---:|
|v2-current|1.2094 → 0.4716|0.62304|30 条：0.71848(step16) → 0.56711(step48)|878.79秒，14.65分|
|A|1.0178 → 0.0025|0.11340|数量24条：0.0091145 → 0.0013522；旧任务30条：1.21164 → 2.07322|721.45秒，12.02分|
|B|1.2332 → 0.2693|0.43767|数量24条：0.35402 → 0.0065737；旧任务30条：0.73996 → 0.49112|1707.36秒，28.46分|

表中首条是首次日志值，不是训练前 loss；A/B 的验证起点是 step16，不是各自训练前值。A 的专项 loss 接近零，同时旧任务验证 loss 上升，是专项过度适配/遗忘风险的直接证据；不能把极低专项 loss 等同为整体最优。

“Qwen→v2→A”与“Qwen→B”均累计96步，确已匹配。但 A 的96条新数据这一阶段跑8个 epoch，B 的新旧混合约4.37个 epoch；A 二阶段重新初始化优化器/调度器，B 连续调度；监督 Token 预算也未严格匹配。因此它们比较的是实际训练策略，不是只改变课程顺序的严格因果消融。

配置证据：`configs/post_train/qwen25_7b_planner_sft_v2_current.yaml`、`count_ab_A.yaml`、`count_ab_B_total96.yaml`。旧 `count_ab_B.yaml` 的48步不是正式B结果；旧B48训练目录当前不存在，不能补造其收敛指标。

## 4. SFT → DPO 的衔接与实际指标

### 4.1 权重与 Reference

- v2：使用已经合并的 v2-current BF16 SFT。
- A：合并 v2-current + A-SFT adapter，得到 A 的 BF16 SFT 父模型。
- B：合并原始 Qwen + B-SFT adapter，得到 B 的 BF16 SFT 父模型。
- 各自的 SFT 父模型以 4-bit 加载，挂一套新的 LoRA 做 DPO；正式训练不是接着5步 smoke权重继续训。
- 已安装 LF 源码 `compute_reference_log_probs` 证实：未另建 ref_model 时，在同一个量化 SFT 模型上 `disable_adapter()`、`no_grad()` 得到参考 log probabilities。参考是各自 SFT，不是原始 Qwen；也没有额外常驻一份完整7B参考模型。
- A/B 的临时合并父目录训练后清理，路径会复用。部署必须按 `logs/shared-dpo-full-20261026-{v2,A,B}-binding.json` 重建对应父权重；不能将 A/B-DPO adapter 直接挂原始 Qwen。

### 4.2 超参数与内存实现

|参数|三组正式 DPO 实际值|
|---|---|
|数据 / epoch / steps|346 对训练、54 对验证；2 epoch；实际88步|
|pref_beta / pref_loss|0.1 / sigmoid|
|pref_ftx|未配置，LF 默认 0，无额外 SFT 辅助项|
|learning_rate|5e-6|
|batch / 累积 / Global Batch|1 / 8 / 8 对，即每完整累积批含16条 chosen/rejected 序列|
|cutoff / quantization|3584 / bitsandbytes 4-bit，NF4、双重量化为默认|
|LoRA / dtype|r16、alpha32、dropout0、all；BF16计算|
|调度 / warmup / optimizer|默认 linear / 0 / adamw_torch|
|验证、保存、选模|epoch策略；44、88步验证；三组最佳checkpoint均88|

实际远端 YAML 中残留 `save_steps:5`、`eval_steps:5`，但正式策略是 `epoch`，因此不能说全量训练每5步验证。beta 是标准 DPO log-ratio 的缩放参数；没有单独测量/汇报 KL，不能把它说成显式加到 loss 上的实测 KL 惩罚值。

`scripts/post_train_low_memory_dpo.py` 在 LF 0.9.3 上对 chosen/rejected 顺序前向、LM head checkpoint、chunk=64 的 FP32 logsumexp 和跳过无用 CE 做低显存处理；存在 `tests/test_post_train_dpo_memory.py`。三组均先通过16对/5步smoke，再各跑约45分钟全量训练。

### 4.3 trainer_state 末步训练指标

下表为各自 **step88 最后一条训练日志**，不是全训练平均，也不是验证平均。最后一个尾批只有2对，不能凭末步100% accuracy证明整体生成质量。

|版本|step|loss|rewards/chosen|rewards/rejected|rewards/margins|rewards/accuracies|
|---|---|---|---|---|---|---|
|v2-DPO|88|0.0887|1.055560|-0.327850|1.383410|1.000000|
|A-DPO|88|0.1092|-0.974031|-1.593577|0.619546|1.000000|
|B-DPO|88|0.0674|-0.085172|-2.098701|2.013529|1.000000|

初始训练loss均约0.6931、reward为0，与新策略从参考SFT初始化一致。全程平均 train_loss 为 v2=0.55415、A=0.52016、B=0.47481。

|54对偏好验证|eval_loss|eval_reward_accuracy|eval_margin|
|---|---|---|---|
|v2-DPO|0.389250|98.148%|1.062854|
|A-DPO|0.339066|98.148%|1.206356|
|B-DPO|0.287834|100.000%|2.091226|

chosen reward 是相对参考模型的 log-ratio，出现负值不等于该答案绝对质量差。A/B 的结果包含更强地压低 rejected 的效应；偏好验证准确率高也不等于完整路线通过率高。

## 5. 推理部署与评测实现

### 5.1 实际加载方式与采样

|版本|评测加载方式|
|---|---|
|Base|原始 Qwen BF16，无 adapter|
|v2-SFT|合并后的 v2-current BF16，无动态 adapter|
|A-SFT|合并 v2-current + 动态 A-SFT LoRA|
|B-SFT|原始 Qwen + 动态 B-SFT LoRA|
|v2-DPO|合并 v2-current + 动态 v2-DPO LoRA|
|A/B-DPO|先临时合并对应 A/B SFT 父模型，再动态挂对应 DPO LoRA|

vLLM 0.11.2、GPU0、BF16、max_model_len=8192、gpu_memory_utilization=0.9、enforce_eager、客户端并发4。4-bit是DPO训练策略，不是本轮vLLM推理精度。

显式请求：temperature=0.3、max_tokens=2048、seed=42/43/44、`response_format=json_schema`、`strict=False`。top_p没有写在评测请求里，但服务日志证实继承模型 generation_config：**top_p=0.8、top_k=20、repetition_penalty=1.05**；配置的默认temperature=0.7被请求0.3覆盖。不能写成top_p=1默认采样。

当前生产Schema hash：`ae6691746c03b337f10b0529f28a0e87fa7ccb9db6aa0ee22bfbc8c583a7c355`；system hash：`402a279c8e218de671211b5c820b88d6cedb181beea8fd92bb7aef2fe4e35c2c`。生产消息由 `capture_messages(state)` 获取，评分见 `build_representative_eval50.py`。

### 5.2 Benchmark与Verifier

正式主比较是5城市×10题=50道开发题，40道常规/10道困难；45道可行、5道应说明不可行。每题三个seed，故每模型每模式150条。**150条不是150道独立题，也不是三个训练随机种子**。

规则覆盖：

1. 候选池成员、必去/排除、跨日去重、天数与天数标签。
2. 严格时间格式、正时长、非重叠、开放/停止入场窗口。
3. 固定道路回放证据、相邻通勤间隔；缺证据单独标错，不真实调用高德。
4. 用户每日数量、游览最小时长、用户时间范围、休息窗口。
5. 38°C及以上时11:00–15:00的室外暴露；室内三值中 unknown 按未证明安全处理。
6. 无请求餐段等约束；不可行分支检查全天空景点、天数正确，以及说明中的原因与关键词，不能仅输出空数组拿分。

生产 `_route_risk_flags` 加 case rubric 评分，并非所有原始风险都一票否决：回放过滤 `opening_time_unknown`、`user_modification`、`long_road_leg` 等标记，`road_unverified` 仍会失败。没有通用老人登阶语义判别器、人工审美/常识评分；不能把规则通过率直接改名为“旅游体验正确率”。不可行题的评测可行性见证不放入输入。

### 5.3 Rules+Rework的真实实现

- Planner-Only：只生成一次。
- Rules+Rework：首轮 + 最多3轮返工，最多4次调用，规则通过即停止；不使用LLM Reviewer。
- 每轮从原始 messages 重新复制，在最后一条 human 内容末尾拼接上一轮错误代码：

```text
\n\n确定性规则返工反馈（仅修正指出的问题，仍须遵守原输入及Schema）：{fault1；fault2；...}
```

- **不把上一轮完整答案或完整历史对话送回模型**，只给错误代码；每轮沿用同一seed。
- 解释一致性 `explanation_faults` 单独计分，但不触发返工。规则通过而解释失败时流程仍会停止。
- 请求异常或无法解析会退出，未实现专门的JSON解析错误修复。旧实现异常时未保存原始响应，是历史归因的证据缺口。

代码：`scripts/run_sft_schema_control.py`、`scripts/run_dpo_eval_50.py`、`scripts/run_shared_prompt_all_versions.py`。当前轮评测已结束，服务此前已关闭；本次审计未重新启动。

## 6. 效果、成本与异常归因

### 6.1 当前同Schema主结果

每格分母150；Delivery在这批记录中按 `parse_success` 操作性统计。所有模型两个模式均150/150、100%，没有另一个独立人工交付质量指标。“规则+解释”额外要求 explanation_faults 为空。

|版本|Delivery P/R|P规则通过|R规则通过|P规则+解释|R规则+解释|无解正确 P→R|R平均返工|
|---|---|---|---|---|---|---|---|
|Base|100% / 100%|103/150（68.67%）|115/150（76.67%）|103/150（68.67%）|115/150（76.67%）|7 → 8 /15|0.813|
|v2-SFT|100% / 100%|111/150（74.00%）|120/150（80.00%）|111/150（74.00%）|120/150（80.00%）|0 → 0 /15|0.680|
|v2-DPO|100% / 100%|112/150（74.67%）|121/150（80.67%）|111/150（74.00%）|119/150（79.33%）|0 → 0 /15|0.700|
|A-SFT|100% / 100%|112/150（74.67%）|122/150（81.33%）|112/150（74.67%）|122/150（81.33%）|11 → 10 /15|0.653|
|A-DPO|100% / 100%|116/150（77.33%）|129/150（86.00%）|114/150（76.00%）|128/150（85.33%）|13 → 15 /15|0.493|
|B-SFT|100% / 100%|122/150（81.33%）|142/150（94.67%）|119/150（79.33%）|138/150（92.00%）|10 → 15 /15|0.260|
|B-DPO|100% / 100%|132/150（88.00%）|142/150（94.67%）|129/150（86.00%）|138/150（92.00%）|14 → 15 /15|0.240|

P=Planner-Only；R=Rules+Rework。无解子集越高越好，但前提是确实不可行且解释正确，不能鼓励任意拒绝。应联合查看135条可行题的交付，避免只优化无解识别。

### 6.2 子项错误分布

下表每格为 **P错误次数 / R最终错误次数**，分母均150，乘100/150即错误率（1次=0.667个百分点）。一条路线可有多个错误，不能求和当失败数。opening_evidence与opening_time_conflict经常是同一条失败的两种记录，不应重复归因。

|错误标签|Base|v2-SFT|v2-DPO|A-SFT|A-DPO|B-SFT|B-DPO|
|---|---|---|---|---|---|---|---|
|commute_time_conflict|2 / 1|2 / 1|1 / 0|20 / 12|17 / 13|2 / 1|0 / 0|
|day_count|1 / 1|0 / 0|0 / 0|9 / 7|4 / 4|0 / 1|1 / 1|
|heat_exposure_or_unknown_indoor|5 / 4|3 / 2|7 / 1|4 / 0|0 / 0|3 / 1|2 / 2|
|infeasible_not_explained|0 / 0|15 / 15|13 / 15|4 / 5|2 / 0|3 / 0|0 / 0|
|infeasible_unsafe_schedule|8 / 7|15 / 15|15 / 15|1 / 2|0 / 0|5 / 0|1 / 0|
|infeasible_wrong_reason|0 / 0|0 / 0|0 / 0|2 / 4|0 / 0|0 / 0|0 / 0|
|minimum_visit|0 / 0|0 / 0|1 / 0|5 / 0|6 / 1|6 / 0|6 / 0|
|opening_evidence|12 / 10|12 / 12|12 / 12|6 / 4|8 / 4|7 / 3|6 / 3|
|opening_time_conflict|12 / 10|12 / 12|12 / 12|6 / 4|8 / 4|7 / 3|6 / 3|
|required_missing|4 / 1|1 / 0|0 / 0|0 / 0|3 / 0|1 / 0|0 / 0|
|road_unverified|1 / 0|1 / 0|0 / 0|0 / 0|0 / 0|0 / 0|0 / 0|
|route_structure|0 / 0|0 / 0|1 / 0|0 / 0|5 / 0|0 / 0|0 / 0|
|unknown_poi|1 / 0|1 / 0|0 / 0|0 / 0|0 / 0|0 / 0|0 / 0|
|user_break_overlap|2 / 1|4 / 0|2 / 1|0 / 0|0 / 0|4 / 1|2 / 1|
|user_time_window|13 / 12|1 / 1|1 / 0|0 / 0|3 / 2|1 / 1|1 / 1|

这张表是最终输出的规则错误，不含额外解释错误。`invalid_time_format`及解析异常在当前14组均为0；解码器受Schema约束，不能据此单独证明模型训练后自主学会了合法时间。

### 6.3 输出长度、Token与端到端耗时

Token为实际API usage；“首轮输出Token”和“每任务输出Token”区分单次生成与所有返工累计。字符数为保存的plan经统一JSON序列化后的首轮平均字符数，**不是原始HTTP响应字符数**；原始字符串未完整保留，不能恢复其精确长度。总Token包含所有输入和输出。

|版本|模式|调用数|首轮输出Token|每任务累计输出Token|首轮规范JSON字符|总输入+输出Token|平均秒/任务|平均返工|
|---|---|---|---|---|---|---|---|---|
|Base|P|150|319.58|319.58|599.93|472590|6.98|0.000|
|Base|R|272|319.93|582.70|600.89|841804|12.01|0.813|
|v2-SFT|P|150|304.48|304.48|633.17|470325|6.55|0.000|
|v2-SFT|R|252|304.11|504.25|632.89|769869|10.39|0.680|
|v2-DPO|P|150|310.24|310.24|640.65|471189|14.31|0.000|
|v2-DPO|R|255|308.77|516.71|638.43|781639|23.47|0.700|
|A-SFT|P|150|257.15|257.15|557.23|463225|11.97|0.000|
|A-SFT|R|248|257.43|402.96|558.17|787381|18.39|0.653|
|A-DPO|P|150|256.49|256.49|556.41|463126|12.32|0.000|
|A-DPO|R|224|256.89|376.11|557.55|709414|17.33|0.493|
|B-SFT|P|150|307.03|307.03|628.19|470707|14.13|0.000|
|B-SFT|R|189|307.25|381.86|628.53|603952|17.24|0.260|
|B-DPO|P|150|311.46|311.46|631.75|471372|14.61|0.000|
|B-DPO|R|186|312.27|388.44|632.33|595410|18.15|0.240|

这些是并发4下客户端端到端耗时，运行批次和动态LoRA加载方式不同，不能严格归因成DPO本身让推理变慢。它们不是独占GPU吞吐率、p95延迟或服务SLA。上述指标为本地模型用量，不产生教师API账单；未取得统一单价和完整历史计费流水，项目总金额不能可靠重算。

### 6.4 B-SFT历史97.3%/96.7%的逐条证据

旧文件：`data/post_train/count_infeasible_ab_20261025/evaluation_total96/regression/B_{planner-only,rules-rework}/results.jsonl`。

|case|seed|Planner-Only|Rules+Rework|
|---|---:|---|---|
|representative50-1-04|43|ValueError: invalid clock|同左|
|representative50-2-04|43|ValueError: invalid clock|同左|
|representative50-4-03|43|ValueError: invalid clock|同左|
|representative50-5-08|44|ValueError: invalid clock|同左|
|representative50-4-08|43|未列为该异常|ValueError: invalid clock|

旧P有4次异常，146/150=97.33%；旧R有5次，145/150=96.67%。9条记录均为 `error_type=ValueError,error='invalid clock',parse_success=false`。代码调用顺序表明响应完成检查和旧TravelRoute结构解析已通过，随后 `score_case → replay_check → _clock_minutes` 抛异常；旧时间Schema没有严格HH:MM正则，因此字段虽是字符串也可能不是合法时钟。

**可确认：评测时钟处理失败，且被混进解析/Delivery失败。不能确认：具体非法时间原文。** 异常记录没有保存原始content或失败plan，无法区分到底是缺前导零、额外文字或其他非法时钟形式。不能描述成已证实的JSON截断、语法错误或字段缺失。后来time_audit的18次重跑没有同样异常，也不能恢复原9次输出。

当前严格时间Schema重测后，B-SFT同样Delivery100%。所以“B-DPO修好了B-SFT的97.3%/96.7%交付”是混用协议后的错误归因；更可信的DPO增益是同Schema下首轮规则通过从122/150升至132/150。

### 6.5 可以成立与暂不能成立的结论

1. **B的混合SFT优于只训专项的A这一实际方案。** B两个验证集loss都下降，最终规则142/150；A专项loss虽更低，但旧任务loss升高，最终规则122/150。
2. **B-DPO主要改善首轮。** 比B-SFT多10条，+6.67个百分点；最终闭环同为142/150，规则+解释同为138/150。不是闭环又多提升一截。
3. v2-DPO两种模式都只多1条，且无解仍0/15；说明仅增加偏好排序，未必能补齐底座缺失的专项输出行为。A-DPO改善闭环7条、无解10→15，但整体仍低于B-SFT。
4. B-DPO相对Base首轮+19.33个百分点、闭环+18个百分点；这属于完整数据/SFT/DPO路径收益，不能全部归给DPO。
5. B闭环总Token603,952→595,410，约减少1.4%；返工0.26→0.24，收益有限。相对Base闭环Token约减少29.3%，但不能据此直接换算线上实际金额。
6. B首轮同题同seed配对18条改善、8条退步，按case分层bootstrap的95%区间约0至13.33个百分点；闭环4改善、4退步，区间约-3.33至3.33。**当前证据不足以宣称稳定显著的未见分布泛化提升。**
7. 50题已经多次用于问题诊断和方案选择，是开发Benchmark。后续应冻结模型与协议，在未参与选模的新城市/来源族留出集一次性比较B-SFT/B-DPO，不再用其反馈补当前训练数据。
8. 历史所谓独立B结果中存在 `count-ab-parent` 实际加载Base的绑定错误；`docs/post_train_independent_B2.md` 和 `post_train_independent_v2_B_compare.md` 已记录失效，不能拿来证明B独立泛化。尚无本次同协议下B-SFT/B-DPO全新留出集结论。
9. 曾中断的4096上下文统一尝试包含大量HTTP失败，不纳入本表；不能将环境/请求失败当模型能力失败或未经日志证明就归因OOM。
10. 当前主表没有同协议教师API组，不能声称已经达到/超过gpt-6.1-sol；教师合成质量与教师独立评测成绩是两回事。

## 附录A：证据导航与复现入口

|模块|真实文件/记录|
|---|---|
|生产结构/消息/风险|`app/planning/schemas.py`、`app/planning/nodes.py`、`app/planning/prompts.py`、`scripts/post_train_runtime.py`|
|当前协议重生成|`scripts/regen_v1_v2_current_protocol.py`；`data/post_train/release_current_protocol_v1_v2/`，含call ledger、训练发布与绑定|
|数量A/B构造与步数匹配|`scripts/prepare_count_infeasible_ab.py`、`scripts/launch_count_ab_matched_steps.py`；`data/post_train/count_infeasible_ab_20261025/`|
|正式SFT YAML|`configs/post_train/qwen25_7b_planner_sft_v2_current.yaml`、`count_ab_A.yaml`、`count_ab_B_total96.yaml`|
|DPO构建/自然失败替换|`scripts/build_shared_dpo.py`、`collect_shared_dpo_pilot.py`、`finalize_shared_dpo.py`、`build_verified_preferences.py`|
|DPO发布|`data/post_train/shared_dpo_20261026/release/{audit.json,manifest.json,shared_dpo_train.json,shared_dpo_validation.json}`|
|DPO正式运行|`scripts/launch_shared_dpo_smoke.py --full`；远端 `logs/shared-dpo-full-20261026-*.log`、`*-binding.json`、`*-results.json`|
|DPO实际YAML|远端 `configs/post_train/shared_{v2,A,B}_planner_dpo_full.yaml`，全文归档于snapshot的actual_yaml；不要用旧通用DPO模板代替|
|真实训练状态|远端 `saves/TravelMind-Planner-v2-current-20261025`、`TravelMind-count-ab-20261025-A`、`TravelMind-count-ab-total96-20261025-B`、`shared-dpo-full-20261026-{v2,A,B}`；各目录trainer_state等在snapshot中|
|统一评测|`scripts/run_sft_schema_control.py`、`run_dpo_eval_50.py`、`run_shared_prompt_all_versions.py`；comparison.json的source指向14组原始结果|
|本次衍生汇总|`data/post_train/engineering_audit_20261027/evaluation_metrics.json`；本报告由同目录build_report.py生成|
|协议与历史异常复核|同目录 `verification.json`；14组消息/Schema/采样协议一致性及9条历史异常摘要|
|推理默认采样的服务日志|同目录 `inference_sampling_evidence.txt`，包含原始日志路径与行号|

未发现/未配置的内容：系统超参搜索、完整历史API货币账单、全部异常响应原文、严格匹配监督Token预算的A/B消融、模型多训练seed重复实验、同协议新独立留出集上的B-SFT/B-DPO比较、当前同协议教师API最终组。不要为面试叙述补造这些实验。

## 附录B：历史迭代如何使用

仓库还保留早期v1/v2、v3、仅迁移提示词MC、v4/v4-clean和大候选池large的配置/报告。这些迭代涉及提示词迁移、答案清洗、训练数据分布及候选池扩容；它们是后续重新生成生产协议答案的背景证据，不是本文当前Schema主比较的额外公平对照。

可沿 `docs/post_train_sft_all_versions_comparison.md`、`post_train_prompt_protocol_audit.md`、`post_train_v2_large_failure_diagnosis.md`、`post_train_sft_v3_result_review.md`、`post_train_migration_control_result.md`、`post_train_quality_v4_result_review.md` 追溯。保留配置不等于相应服务器权重/训练日志仍在；对于缺失的历史训练状态，本报告不从静态YAML推测实际loss。最终面试主线建议按“协议审计 → 规则可证数据 → 数量专项持续训练与混合训练对照 → 同SFT参考的共享DPO → 同Schema首轮/闭环评测”陈述，并明确开发集和统计局限。
