# 2026-10-24 数据审计与试训放行

## 已完成

本轮自动复检与代理抽查，不等同于独立人工专家验收。检查真实首轮消息是否与生产构造一致、原始输出Schema、候选范围、时间/道路证据、显式偏好、引用、首轮答案中的返工上下文泄漏。抽查包30条的需求、解释与路线已阅读，发现两条额外引用未提供的前版方案，连同规则识别的6条隔离。194条候选保留186条；原始数据不删除。

按城市与完全相同候选名称集合分组后，156条训练、30条验证，分组交集为0。仍有城市与候选地点重叠，不能称独立城市测试。丽江/景德镇留出资产缺道路证据，尚未就绪。

## 发现的问题与处理

- 人工负例常修改路线但保留正例解释，DPO可能只学解释矛盾。暂不导入正式偏好数据，原始记录保留。
- 首轮监督不能混入“上一版/取消原路线”等依赖返工上下文的答案，8条隔离。
- 候选室内标签部分由类别推断，特别是总统府等复合景区，不能证明全程室内；已通过规则不代表现场天气安全。
- 海岛游览的道路快照不完整覆盖乘船、排队、末班船，解释中有风险披露，但独立运营事实未核验。不可宣称这些数据已证明全链路可执行性。
- 当前天然偏好对仅6对训练/1对验证，足以测试优化流程，远不足以证明DPO质量或支持正式DPO结论。

## 放行范围

仅允许SFT/DPO工程smoke，SFT最多16条、DPO实际只有6条（max_samples=16是上限，不能说有16对），各5步。样本按Qwen tokenizer检查，最长2191，小于3584，不静默截断。完整训练尚未放行；需要进一步解决偏好数据质量和独立测试证据。

训练环境固定Python3.10/LLaMA-Factory0.9.3、Transformers4.52.4、PEFT0.15.2、TRL0.9.6、Accelerate1.7.0，复用torch2.9.0+cu128。全部新包在本机下载后上传/home/jiale，独立venv覆盖训练包，未修改vLLM的用户包。使用SDPA避免依赖未安装的FlashAttention2扩展，显存和吞吐必须实测。


## GPU0 实测结果（2026-10-24）

SFT 5步成功：峰值allocated 18.1045 GiB，reserved 21.8281 GiB，训练loss 1.6071。4-bit QLoRA DPO首次因完整词表概率及参考前向显存不足失败，分块计算后仍在反向阶段OOM；最终采用逐序列前向、完整评分路径checkpoint、分块FP32 logsumexp，去除未使用的模型内CE损失，保留sigmoid DPO和SFT参考模型。最终5步成功：峰值allocated 18.7282 GiB，reserved 19.4453 GiB，训练loss 0.6911，训练14.20秒、加载及保存共21.08秒。6对数据不能支持偏好胜率结论，5步loss也不能证明业务收益。

实现保存在scripts/post_train_low_memory_dpo.py，针对LLaMA-Factory0.9.3启用，未修改站点包或生产推理逻辑。CPU测试验证FP32/BF16 completion概率及梯度，以及逐序列checkpoint的模型参数梯度与原始完整log_softmax实现一致。当前真实样本最长2191 tokens；3584仅为配置上限，尚未完成满长显存压力测试。SFT与DPO smoke权重不作为正式发布模型。

下一阶段按顺序：1）完成3584长度压力测试；2）审核自然偏好对是否确实更差，优先构造路线和解释一致、规则可证的单缺陷负例并复检；3）补齐独立城市留出集道路/开放窗口/可行解证据，无法证明的场景不得计为通过；4）小规模SFT学习曲线与验证集对照，再决定是否扩量；5）统一四组模型同输入、同规则、同失败分母，报告首轮硬约束、软约束、完整成功率、返工/耗时/token，独立集单列并给出区间。当前阶段不新增付费API请求。

原始显存记录已回传data/post_train/release_v1/gpu_training_metrics.jsonl，DPO日志同目录dpo-smoke.log。恢复GPU0原始Base vLLM服务后进行真实请求验证。


### 本轮最终验收

本地pytest：553 passed，3项既有弃用警告。vLLM恢复时补齐CPATH，复用/home/jiale/python-dev/extracted内已有Python3.10开发头文件；启动脚本scripts/serve_post_train_4090.sh可复现，不需要新增下载。GPU0服务真实JSON请求成功。进一步使用真实Joint Planner消息和TravelRoute JSON Schema调用Base模型：1152输入token、368输出token、6.47秒，Pydantic解析成功。该单例原始输出未通过业务规则（meal_unknown_restaurant、road_unverified、route_structure），因此这里只验收接口/Schema连通，绝不计为规划成功或四组正式指标。请求、原始响应、规则结果保存在release_v1/planner_service_smoke.*。未修改模型服务为smoke权重。
