## 2026-10-24 16:38 第一轮正式训练已启动

使用全部156条SFT训练样本、30条验证、3epochs、30个优化步、梯度累积16、seed42，在GPU0运行。不是16条5步smoke。每轮验证并保存检查点，以最低eval_loss选择最终适配器，随后CPU合并；不使用城市留出集选模型。已有审核文件中full_training_release:false是当时历史审计状态，本轮用户明确授权启动全合格数据训练，独立运行清单记录授权与输入哈希，不伪造人工认证。

采取SFT→30条开发验证Base/SFT对照→检查退化后决定DPO。run_post_train_4090.sh --sft-only不自动接DPO，更新DPO配置为preferences_v2的154对训练/30对验证，防止误用旧6对数据。训练结束自动恢复原Base vLLM，随后本机后台对照任务临时加载SFT，评测后再恢复Base；原始请求、响应、usage、逐样本错误与配对改善/退化保存data/post_train/sft_validation_v1，报告docs/post_train_sft_validation_report.md。零付费API调用，不读取独立测试可行解。

16:41观察：13/30步，第一轮验证loss1.2063，显存约20.3GiB，无OOM。GPU0温度后续升至87°C并触发硬件的软件热限速，普通用户无权降低功耗上限，未更改系统功耗配置；因此完成时间按实测速率动态估计，而非按smoke外推。初估训练+合并16:47–16:50，开发验证对照另约5–10分钟，尚非最终四组测试。

本轮代码回归564项pytest通过，3项既有弃用警告。


### 16:45 训练结果

正式SFT的30步/3epochs已完成，无OOM。每轮验证loss为1.2063158、1.0114537、0.9620780；最佳checkpoint-30。加载训练验证及保存419.69秒；PyTorch显存峰值allocated18.9794GiB、reserved19.8340GiB。GPU0全卡nvidia-smi读数约20.3GiB，包含PyTorch以外开销，两种口径不同。权重CPU合并正在进行；DPO尚未启动。loss下降不等于规划成功率提升，仍待30例开发验证Base/SFT对照。


### 16:48 首十分钟观察结束

SFT训练及CPU合并已成功完成，总耗时470.40秒。GPU0训练时86–88°C曾热限速，推理评测时回落63°C。Base/SFT开发验证任务已在后台执行，DPO不会自动启动；初始已完成Base结果均保留，包括不合法时间格式导致的评分失败，不删除失败或当API故障。预计16:55–17:00完成开发验证及恢复Base服务。按用户要求停止持续监视，后台任务继续落盘；状态文件data/post_train/sft_validation_v1/job_status.json，原始结果base.jsonl/sft.jsonl。
