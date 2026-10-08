# RAG v2 实施与实验记录（2026-10-08）

本次交付评测工具、采集快照、待审标注及开发集诊断，**尚未达到正式评测完成条件**。
生产默认检索未改变，未运行付费生成或 Judge，不宣称检索效果提升。

## 语料与标注

结构校验通过：六城 128 块，南京 23、上海 20、重庆 23、丽江 20、三亚 21、景德镇 21。
其中 28 块为项目作者的规划建议，明确与官方事实分开；丽江仅 7 块采集内容、13 块建议，
因此不能写成“每城 20 块官方事实”。来源有效性、时效性仍需人工核验。

语料指纹：`5bf231483310c9c74108c2a0d31491e69f8e24c13942aa8960103d82e7356d1f`。
原始快照、排除原因及近似重复别名保留在 knowledge 的版本目录与 manifest 中。

120 条查询为机器提出的**待审草稿**：90 条可回答候选、15 条缺失事实候选、15 条范围外候选；
开发/测试各 60 条，同意图改写未跨集合。已经去掉把 API 路径当景点名的模板问题，加入口语、
跨城市干扰和城市消歧。真正的同名景点、必要事实、多证据标签及可回答分类仍需要人工修订，
不能把这些候选分类视为已验证的数据集质量。

标注工具支持导出、导入查询、导入来源审核和冻结。正式冻结同时检查来源签名、查询签名、
配置有效性及实际 tokenizer 硬上限；修改任一绑定内容会使冻结失效。

## 已运行实验与限制

`evaluation/rag_v2_dev_sweep_draft.json` 保存六种检索方式、k=1/2/3/5/8/10 和融合参数扫描的
逐请求结果、错误及指纹。关键词、BM25 的 60 条开发查询运行成功；向量、BM25+向量 RRF、
关键词+向量 RRF 和旧融合方式的每组 60 条均因本地缺少固定模型快照失败。
这些失败被计入总体质量与可用率，成功请求质量单独保留，没有退回关键词掩盖失败。
草稿 k 候选不是默认配置建议，更不能据此宣称 BM25 优于向量。

模型选择：BAAI/bge-small-zh-v1.5，revision `7999e1d3359715c523056ef9478215996d62a620`。
首次实验时模型依赖已安装，但 Hugging Face 的权重/tokenizer 下载连接超时；直接读取固定 revision
的 tokenizer 配置也超时。评测查询只读本地缓存，下载通过 prepare_rag_embedding 显式执行。
MiniLM 支持显式固定 revision 的独立诊断配置，未运行该诊断。

已验证可以按采集顺序恢复 122 个来源段落，保留文本总字符数不变；分块对照工具已经改为
从这些段落出发。首次实验未能生成实际 tokenizer 的两套分块；同日网络重试的结果见下节，
不能用模拟 tokenizer 测试结果替代真实分块验收。

BM25 单查询本地时延诊断（“上海博物馆怎么预约”，同一 128 块语料）：独立进程冷启动 3 次，
预热后 10 次。冷 P50/P95 为 543.16/569.03 ms，暖 P50/P95 为 0.195/0.301 ms。
原始观测见 `evaluation/rag_v2_bm25_latency.json`；仅单查询诊断，**不是线上 SLA 或优化比例**。

生成预检见 `evaluation/rag_v2_generation_preflight.json`：60 测试请求 × 3 个 k × 3 次重复 ×
生成/Judge = 最多 1080 次调用。示例输入/输出单价 1/2 每百万 Token 只是命令示范；
当前预估上限 31.6206 个用户指定货币单位，不是供应商报价或实际费用。未执行付费调用，
输出质量字段为 null。工具保留实际 usage、耗时、失败、每 k/类型汇总和人工抽查列表，
调用前预留预算，超预算停止；Judge 不替代人工核查。

## 验收与复现

已运行旧评测资产校验、v2 配额/分组校验、编译检查和完整 pytest：
**346 passed，32.78 秒，3 条依赖弃用警告**；RAG 专项 **49 passed**。
专项测试覆盖指标公式与去重、失败计分、BM25 缓存、索引内容修复、模型切换、分块超限、
用户过滤、空输入、严格混合检索、冻结变更保护、生成预算和未知 usage。
测试集 CLI 在待审标注下正确拒绝生成正式报告。

复现命令、审核字段和逐阶段操作见 [rag_benchmark_v2.md](rag_benchmark_v2.md)。
运行时 data 目录不纳入 Git。

## 尚需完成

1. BGE 下载、真实分块及开发诊断已完成（见下节）；MiniLM 诊断对照尚未运行。
2. 人工核验来源、修订 120 条查询的事实/证据组，尤其补齐同名景点与真实多证据问题。
3. 用人工确认后的开发集选配置，再签名冻结一次性测试集评估；当前候选配置未测定最优。
4. 确认真实模型价格与付费预算后，运行三次重复生成及独立 Judge，完成抽样人工核查。
5. 仅在以上证据与复现结果支持时调整生产默认，提交冻结报告；本次不提前执行这两阶段。

## 同日网络重试与真实 BGE 诊断

使用本机代理后小文件下载成功，但整份权重传输曾在约 17 MB 中断。改为可重试的 HTTP Range
分段下载，最终获得 95,827,648 字节；整份权重 SHA256 与固定 revision 的上游 LFS 哈希一致：
`354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026`。
本地 `prepare_rag_embedding.py` 验证 `model_loaded=true`，输出向量形状 `[1, 512]`。
权重保存在用户 Hugging Face 缓存，不提交模型或运行时索引。

`knowledge/benchmark_v2_variants/` 保存同源的实际 tokenizer 分块与重新映射的待审标签：

| 分块 | 块数 | 最大 Token（含特殊符号） | 硬上限 |
|---|---:|---:|---:|
| 段落/句子语义分块 | 128 | 400 | 512 |
| Token 边界分块 | 126 | 508 | 512 |

两套语料的城市配额、唯一 ID、哈希及查询引用校验均通过。原段落计数时的超长警告不代表
最终分块截断；最终每块都已按真实 tokenizer 检查。每套仍是待审标签，不是人工金标准。

分别在 60 条开发查询上运行六种检索方式、六个 k，以及融合 c=20/60/100，得到每套 60 个
配置结果。两套均无请求失败，保留逐请求排名、指标、指纹与分组 bootstrap：

- `evaluation/rag_v2_bge_semantic_dev_sweep_draft.json`
- `evaluation/rag_v2_bge_token_dev_sweep_draft.json`

以下仅展示语义分块 k=3、c=60 在 **45 条待审可回答查询**上的诊断，不用于简历或正式结论：

| 检索方式 | Recall | Hit | MRR | nDCG |
|---|---:|---:|---:|---:|
| 关键词 | 0.080 | 0.156 | 0.100 | 0.079 |
| BM25 | 0.093 | 0.111 | 0.070 | 0.073 |
| 向量 | 0.171 | 0.200 | 0.152 | 0.148 |
| BM25+向量 RRF | 0.120 | 0.178 | 0.126 | 0.113 |
| 关键词+向量 RRF | 0.160 | 0.200 | 0.130 | 0.128 |
| 旧融合 | 0.160 | 0.200 | 0.130 | 0.128 |

草稿得分较低，不能把“模型可加载”当作“检索质量合格”。所有模式的草稿候选 k 都为 10，
优先核查标签与事实覆盖，不能据此直接扩大生产召回。混合检索相对 BM25 的 Recall 差值
为 0.0267，意图组 bootstrap 区间为 [0, 0.0643]，不足以建立稳定提升结论。
草稿改善案例包含南京博物院老人出行、公共设施与景德镇雨天安排，逐请求证据保留在 JSON。

15 条待审无答案查询（缺关键事实 8、范围外 7）均仍返回检索结果。当前没有标定拒答阈值，
这暴露了需要后续验证的无答案处理，不能宣称拒答有效，更不是生成幻觉率。

真实 BGE 混合检索单查询时延诊断：独立进程冷启动 3 次，预热后 10 次，
冷 P50/P95=5561.33/5894.76 ms，暖 P50/P95=16.17/28.65 ms，错误数 0。
原始数据见 `evaluation/rag_v2_bge_hybrid_latency.json`。冷启动包含解释器、导入、模型加载与查询；
仅 CPU 本地单查询，非线上 SLA，也不能与先前 BM25 时延直接推导优化比例。

复现新增实验：

```powershell
python scripts/prepare_rag_embedding.py
python scripts/sweep_rag_v2.py --corpus knowledge/benchmark_v2_variants/semantic.json --cases knowledge/benchmark_v2_variants/semantic.cases.json --out evaluation/rag_v2_bge_semantic_dev_sweep_draft.json
python scripts/sweep_rag_v2.py --corpus knowledge/benchmark_v2_variants/token.json --cases knowledge/benchmark_v2_variants/token.cases.json --out evaluation/rag_v2_bge_token_dev_sweep_draft.json
python scripts/benchmark_rag_latency.py --corpus knowledge/benchmark_v2_variants/semantic.json --query "上海博物馆怎么预约" --mode hybrid --out evaluation/rag_v2_bge_hybrid_latency.json
```

本次重跑评测资产校验、两套分块配额/标注引用校验和 RAG 专项：49 passed。
没有运行冻结测试集、付费生成或 Judge；生产默认不变。
