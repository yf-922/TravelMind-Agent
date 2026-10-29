"""Summarize a completed frozen prompt screen without additional model calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.compare_planner_prompt_candidates import OUT, PROMPTS, ROOT, report, rows


def main():
    status = json.loads((OUT/'status.json').read_text(encoding='utf-8'))
    if status['stage'] != 'completed':
        raise RuntimeError('Experiment must finish before drawing conclusions')
    data = rows()
    if len(data) != 144:
        raise RuntimeError('Expected all 144 paired generations')
    paid = [r for r in data if r['backend'] == 'online']
    combined = {}
    for name in ['current', 'checklist']:
        combined[name] = {}
        for backend in ['online', 'v2', 'large']:
            group = [r for r in data if r['prompt']==name and r['backend']==backend]
            combined[name][backend] = dict(n=len(group), joint=sum(r['joint_passed'] for r in group))
    paired = {}
    for backend in ['online', 'v2', 'large']:
        a = {(r['phase'], r['id']):r for r in data if r['prompt']=='current' and r['backend']==backend}
        b = {(r['phase'], r['id']):r for r in data if r['prompt']=='checklist' and r['backend']==backend}
        paired[backend] = dict(
            current_only=sum(a[k]['joint_passed'] and not b[k]['joint_passed'] for k in a),
            checklist_only=sum(b[k]['joint_passed'] and not a[k]['joint_passed'] for k in a))
    evidence = dict(
        recommendation='Retain current as provisional shared baseline; checklist is a v2 challenger, not a proven universal improvement.',
        production_changed=False, combined=combined, paired=paired,
        paid_calls=len(paid), paid_reported_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in paid),
        elapsed_generation_minutes=(max(r['started_at']+r['elapsed_s'] for r in data)-min(r['started_at'] for r in data))/60,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'scripts/compare_planner_prompt_candidates.py', 'app/planning/prompts.py',
            'scripts/build_representative_eval50.py', 'scripts/prepare_sft_v2_release.py']},
        protocol_warning='Screen and confirmation are development cases, single generation per model/case/prompt; pooled counts are descriptive after selection, not an independent significance test.')
    (OUT/'decision.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report()
    path = ROOT/'docs/post_train_prompt_screen.md'
    original = path.read_text(encoding='utf-8')
    intro = '''# Joint Planner 提示词筛选结论

本轮已经真实完成，暂选 **current（当前生产版）作为线上、未来训练和评测的共享协议基线**，不宣称它已最优。执行检查版 checklist 保留为挑战候选。本轮没有修改生产提示词、模型权重、训练集或冻结评测资产。

选择理由：预先固定的首轮排序中 current 第一；但新增确认题上 checklist 更好，合并后两者均32/42，证据不支持直接换提示词。模型间有权衡，不能只看总体分数。本轮候选不是越短越好，也不能通过多写规则保证成功。

主指标为首轮零返工的“规则通过＋已有确定性解释检查通过”；并非人工对全部语义的认证。

|首轮10题|字符数|线上 gpt-6.1-sol|7B SFT-v2|7B v2-large|合计|
|---|---:|---:|---:|---:|---:|
|current 当前生产版|1100|10/10|6/10|9/10|25/30|
|compact 精简版|417|8/10|5/10|6/10|19/30|
|balanced 分层约束版|732|9/10|5/10|4/10|18/30|
|checklist 执行检查版|698|9/10|7/10|7/10|23/30|

|另4题确认|线上|SFT-v2|v2-large|合计|
|---|---:|---:|---:|---:|
|current|4/4|1/4|2/4|7/12|
|checklist|4/4|3/4|2/4|9/12|

|两阶段合并（仅描述）|线上|SFT-v2|v2-large|合计|
|---|---:|---:|---:|---:|
|current|14/14|7/14|11/14|32/42|
|checklist|13/14|10/14|9/14|32/42|

## 为什么这样设计及如何解释

四版使用相同当前生产构造的human消息，只替换system；同一Schema、候选池、道路回放和代码rubric。线上模型是受测对象，不是裁判。10题覆盖5城市、10类问题，另4题不参与首轮候选排序；全部来自现有开发套件，不能称为独立泛化测试。每题每模型每提示词只生成一次，合并分数含筛选后偏差，不作显著性结论。

compact压缩关键规则；balanced按优先级/选点/时间/不可行/输出分层；checklist按执行步骤核对时间、去重与解释。三版尝试把历史发现的问题讲清楚，但没有在本次实测中形成共同优势。原因不能仅凭这14题归结为“注意力涣散”，还可能包括训练提示词熟悉度、字段说明完整度和采样波动。

本地v2首轮平均输入Token为：current 2746.9、compact 2457.9、balanced 2638.9、checklist 2653.9。checklist比current仅节约93个输入Token（约3.4%），因为候选池与human上下文占据大部分输入。字符数不能直接等同于Token数或推理负担。最短版也没有带来更可靠的结果。

## 剩余缺陷与证据边界

1. current仍有天数/每日数量、高温边界、不可行时强排等失败；checklist仍有开放时间、高温、天数和解释引用问题。提示词不能替代本地规则校验和返工。
2. v2的compact有1次、balanced有3次达到2048输出Token而截断，均计失败，不补齐、不重试。这是本输出预算下的行为，不能推断更高预算必然失败。
3. 线上代理接受strict Schema参数，但compact、balanced各1次实际输出未满足Schema，计失败。不能仅凭参数被接受便认为约束生效；仍需字段说明和客户端校验。
4. 本轮schema_parse指标检查JSON/Schema；period语义、非法钟点、天数和时间冲突另计规则失败。没有通过后处理修复格式抬高成绩。
5. 与历史150题结果不可直接比较：本轮只有14个开发题，system与结构化接口设置也不同。尤其不能凭v2-large在这里9/10便宣布其历史问题已解决。
6. 当前只验证了首轮；餐馆、检索注入、复杂新增需求等未单独充分覆盖。未做完整语义人工评审，未证明多seed鲁棒性或真实线上满意度。

## 后续统一协议的执行建议

先把current作为暂定共享基线，system全文见下方及同目录current.txt。需要冻结的是完整messages构造函数、Schema、天气边界、属性语义和解码配置，不只是system字符串；实际城市/天气/候选内容可以随输入变化。知识开关等须一同记录。过去权重见过的旧提示词无法通过修改JSON追溯改变。

下一项最有价值的验证是current/checklist在现有50题上的同题多seed本地配对对照，先不再付费调用教师；关注每天数量、不可行识别、高温/开放窗口、截断和解释一致性。报告按城市和难度分层，保留原始失败，不只报告综合平均。它仍是开发选型，最终需另留未用于提示词选择的独立题集。

如果确定部署v2，checklist值得继续验证；若要求同一提示词适配线上和多个7B版本，当前数据不足以支持切换。不要用不同提示词分别调出各模型最高分后再当作训练版本的公平对照。确定最终协议后，新合成/训练/评测/线上共同使用并记录哈希；本轮没有触发重新合成或训练。

'''
    intro += f"本轮共144次生成，其中线上{len(paid)}次、代理报告{evidence['paid_reported_tokens']:,} Token（非价格账单）；真实生成阶段约{evidence['elapsed_generation_minutes']:.1f}分钟。新增测试5项已通过。原始messages、输出、usage和判定保存于`data/post_train/prompt_screen_20261025/results.jsonl`，冻结协议见`binding.json`，汇总及代码哈希见`decision.json`。\n\n---\n\n"
    path.write_text(intro+original,encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
