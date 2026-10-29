# 候选池与有界 ReAct：离线配对消融

## 测量范围与限制

28 个主动构造的检索后缺口：9 点名、9 室内、5 数量、5 边界；覆盖南京、上海、三亚。不是随机真实用户样本，不报告置信区间或线上成功率。
两组从完全相同的固定基础候选池出发，Baseline 不补查；实验组运行生产 LangGraph 决策包装器→搜索→校验→条件边。LLM 决策由读取当前缺口的确定性策略替身代替，因此不验证真实 LLM 的工具选择能力。
POI 来自已有真实高德响应快照，保留原名称、ID、坐标、类型与评分。点名查询按同名过滤原始记录，分类查询返回冻结博物馆记录；这是受控 HTTP 响应映射，不是这些具体查询的实测响应或实时高德召回率。空结果、超时、重复行、近名干扰为显式注入。
基础池来自旧快照，其中室内标签是历史名称启发式。新增 POI 严格走当前生产解析器，分类推断只提供 indoor_hint，不会升级为确定 indoor。室内结果只能在此标签口径内解释，未人工核实。
生产基础检索本身已按请求选择查询词；本实验刻意选取其后仍有缺口的状态，不代表完整基础检索平均表现。4 次 HTTP 是共享预算：多数场景起点为 0（前置开销未录制），另有起点已用 3 次的共享预算探针；报告同时保留初始、补查、总次数，不声称全流程实测最多 4 次。
候选前提独立评分：点名集合全部出现、去重数量达标、明确室内数量达标，三项同时满足才通过。生产 validator 状态另外记录；不执行 Planner，不将前提满足冒充最终路线可规划/成功率。主分母为 23 个非边界受控缺口，不称为全都可满足；其中可能存在固定响应容量不足，另报诊断子集。
无真实 LLM、外网或付费调用；禁用缓存、退避等待。HTTP 重试与预算使用生产 Provider 实现。不存在景点只在受控响应中设为空，不声称现实中同名地点一定不存在。

## 结果

| 场景 | 仅基础候选 | + 有界补查 |
|---|---:|---:|
| 点名缺口 | 0/9 (0.00%) | 9/9 (100.00%) |
| 室内缺口 | 0/9 (0.00%) | 0/9 (0.00%) |
| 数量缺口 | 0/5 (0.00%) | 4/5 (80.00%) |
| 非边界缺口合计 | 0/23 (0.00%) | 13/23 (56.52%) |

- 基础候选前提受阻：23/23 (100.00%)；补查后仍受阻：10/23 (43.48%)。
- 边界安全退出并保留不足：5/5 (100.00%)；轮次/预算合规：28/28 (100.00%)。
- 虚构/近名错误接纳：0/2 (0.00%)；重复查询拦截：6/6 (100.00%)（决策侧独立计算重复动作数作分母，核对实际 HTTP 未重复）。
- 补查共 45 轮，均值 1.6071 轮/单；HTTP 尝试共 51 次，均值 1.8214 次/单；最大 2 轮、总 HTTP 4 次。
- 执行异常：0/28 (0.00%)，失败请求不剔除。
- 若误把 indoor_hint 当作确定室内，诊断得分会变成 20/23 (86.96%)；该数不是正式通过率，不能用于宣称室内安全。

- 即使忽略动作预算并把全部类型提示当作室内，固定响应仍不足的场景：indoor-3-1, indoor-3-2, count-3。未将这些失败悄悄删除。
- 在上述乐观假设下固定响应容量足够的诊断子集：基础 0/20 (0.00%)，补查 13/20 (65.00%)；最终评分仍遵守正式 indoor，不用提示替代事实。

## 逐例结果

| ID | 需求 | 基础→补查 | 轮次/HTTP补查/初始 | 状态与未满足前提 |
|---|---|---|---|---|
| named-1-1 | 南京一日游，想去“南京博物院” | 0→1 | 1/1/0 | ready:  |
| named-1-2 | 南京一日游，想去“南京地质博物馆”，想去“南京市博物馆” | 0→1 | 1/2/0 | ready:  |
| named-1-3 | 南京一日游，想去“江宁织造博物馆” | 0→1 | 1/1/0 | ready:  |
| indoor-1-1 | 南京雨天游，室内优先 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| indoor-1-2 | 南京只去室内 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| indoor-1-3 | 南京雨天游，室内优先 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| named-2-1 | 上海一日游，想去“上海博物馆(人民广场馆)” | 0→1 | 1/1/0 | ready:  |
| named-2-2 | 上海一日游，想去“上海自然博物馆”，想去“上海邮政博物馆” | 0→1 | 1/2/0 | ready:  |
| named-2-3 | 上海一日游，想去“中国证券博物馆” | 0→1 | 1/1/0 | ready:  |
| indoor-2-1 | 上海雨天游，室内优先 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| indoor-2-2 | 上海只去室内 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| indoor-2-3 | 上海雨天游，室内优先 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| named-3-1 | 三亚一日游，想去“三亚山海贝文化博物馆” | 0→1 | 1/1/0 | ready:  |
| named-3-2 | 三亚一日游，想去“三亚李是亨黄花梨博物馆”，想去“海南京润珍珠博物馆” | 0→1 | 1/2/0 | ready:  |
| named-3-3 | 三亚一日游，想去“绿盾高性能系统门窗” | 0→1 | 1/1/0 | ready:  |
| indoor-3-1 | 三亚雨天游，室内优先 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| indoor-3-2 | 三亚只去室内 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| indoor-3-3 | 三亚雨天游，室内优先 | 0→0 | 2/2/0 | insufficient: indoor_not_established |
| count-1 | 南京历史与自然结合，候选去重筛选后数量不足，需要补充 | 0→1 | 2/2/0 | insufficient:  |
| count-2 | 上海历史与自然结合，候选去重筛选后数量不足，需要补充 | 0→1 | 2/2/0 | insufficient:  |
| count-3 | 三亚历史与自然结合，候选去重筛选后数量不足，需要补充 | 0→0 | 2/2/0 | insufficient: insufficient_count |
| count-4 | 南京历史与自然结合，候选去重筛选后数量不足，需要补充 | 0→1 | 2/2/0 | insufficient:  |
| count-5 | 上海历史与自然结合，候选去重筛选后数量不足，需要补充 | 0→1 | 2/2/0 | insufficient:  |
| nonexistent-repeat | 想去“霍格沃茨魔法学校” | 0→0 | 2/1/0 | insufficient: missing_required_poi |
| near-name-reject | 想去“南京博物院魔法分馆” | 0→0 | 2/1/0 | insufficient: missing_required_poi |
| round-limit | 雨天只去室内 | 0→0 | 2/4/0 | insufficient: indoor_not_established |
| retry-budget | 只去室内 | 0→0 | 1/4/0 | insufficient: indoor_not_established |
| shared-budget | 只去室内 | 0→0 | 1/1/3 | insufficient: indoor_not_established |

## 结论与下一步

点名补查和数量补查在固定响应下能够修复部分候选前提，证明补查工具链有实际作用；该结果不能证明真实模型每次能选对动作，也不能证明最终行程成功。
室内组未改善：新增馆类 POI 的 indoor 仍为未知，检索返回不等于取得室内证据。三亚另有评分过滤后响应容量不足。应先补可追溯的室内证据与候选质量诊断，再评估是否需要额外补查策略；本次未修改生产规则以追求更高分。
零分基线是选样条件：本集专门挑选基础池存在缺口的状态；不应写成基础检索总体通过率 0%，也不应把 5 条边界的安全退出率当成全系统可靠性。
本次只新增评测工具、版本化资产、回归测试和报告；生产候选检索实现及默认配置未变。

## 复现与审计

```powershell
python scripts/evaluate_candidate_react.py
python -m pytest -q tests/test_candidate_react.py tests/test_candidate_react_benchmark.py
python scripts/validate_eval_assets.py
python -m pytest -q
```

逐请求 JSON：`docs/candidate_react_ablation.json`，包含两组评分、缺口、候选名称、决策观察、搜索追踪、HTTP 尝试和停止状态。测试与报告均使用版本化快照，复现不依赖被忽略的 data/。没有测量 Token、网络耗时或真实模型质量。

指纹（SHA256；数据按排序键规范化 JSON）：

```json
{
  "snapshot": "92ce86aa114ffb2897c4f84070783f7ed64ec2cb0edf48780b4677a2cb72f820",
  "cases": "a3ae6b8212de5e8756eeb1f03d9aedf5dcbd5e46ac3b9cc453b77d4860af5490",
  "config": "147c0aae6bc0f73c6959c04974e83010232e52681479235c005ab26c2e1abdc1",
  "code": {
    "app/planning/candidate_react.py": "18c50ecf8121c6822cc44636963b8284e506e7c39ad7bc25ce9933c1e4373c3b",
    "app/planning/schemas.py": "a1c639f2db0ef301706235cb485ab209b1c05209b3c2e7bfef8e0a8b8c83009d",
    "app/planning/helpers.py": "ae2a3ed6be9c7510b0e8348298a7b005d87851b6a129492d47ce7d282dc1f7e2",
    "app/providers/amap/poi.py": "51f350a52774ccc446e56c780d7fb1ed63c3546ee85026447002bac3c951a59b",
    "scripts/evaluate_candidate_react.py": "641f5bcd3532ad868aaeec6927ce52ad4406e23845344d72543ec17044c76b88"
  }
}
```
