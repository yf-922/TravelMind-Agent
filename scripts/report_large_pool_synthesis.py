"""Read-only progress report for the versioned fresh-teacher experiment."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_sft_v2_release import read_rows


def main():
    folder=ROOT/'data/post_train/v2_large_pool_teacher_contract_v3'
    status=json.loads((folder/'status.json').read_text(encoding='utf-8')) if (folder/'status.json').exists() else {}
    rows=read_rows(folder/'synthesis.ledger.jsonl') if (folder/'synthesis.ledger.jsonl').exists() else []
    accepted=[r for r in rows if r['status']=='accepted']
    inputs=json.loads((folder/'input_audit.json').read_text(encoding='utf-8'))
    all_calls=[]
    global_calls=0
    for path in (ROOT/'data/post_train').rglob('*.calls.jsonl'):
        records=read_rows(path)
        global_calls+=sum(r.get('event')=='started' for r in records)
        if 'v2_large_pool_teacher' in str(path):
            all_calls.extend(records)
    usage=Counter()
    for r in all_calls:
        if r.get('event') in ['completed','failed']:
            for k,v in (r.get('usage') or {}).items():
                if isinstance(v,int):usage[k]+=v
    starts=sum(r.get('event')=='started' for r in all_calls)
    unmetered=sum(r.get('event')=='failed' and not r.get('usage') for r in all_calls)
    dates=Counter(r['state']['days'] for r in accepted)
    text=(f'# 扩池教师答案合成进度\n\n更新时间：{datetime.now().astimezone().isoformat()}。\n\n'
        '当前生产 Joint Planner 已明确纯 JSON、固定字段、仅输入快照、禁止工具调用和简洁解释；'
        '通过真实生产函数捕获消息。旧答案和评测答案没有提供给教师。未启动训练或 DPO。\n\n'
        '|项目|数量|\n|---|---:|\n'
        f"|母本输入|{inputs['source_inputs']}|\n|可生成输入|{inputs['callable']}|\n|输入证据隔离|{inputs['isolated']}|\n"
        f"|已完成记录|{len(rows)}|\n|新审计下接受|{len(accepted)}|\n|剩余输入|{inputs['callable']-len(rows)}|\n"
        f"|当前版本实际调用|{status.get('calls',0)}|\n|整个扩池实验实际调用|{starts}|\n|全局累计实际调用|{global_calls}|\n"
        f"|扩池实验接口报告总 Token|{usage['total_tokens']}|\n\n"
        f'缺少用量响应的失败调用：{unmetered} 次，已计入调用预算；报告Token是已返回部分，不能据此断言失败调用零成本。\n\n'
        '输入隔离原因：'+json.dumps(inputs['isolation_reasons'],ensure_ascii=False)+'。\n\n'
        '接受数据城市分布：'+json.dumps(dict(Counter(r['state']['destination'] for r in accepted)),ensure_ascii=False)+'。\n\n'
        '接受数据天数分布：'+json.dumps(dict(dates),ensure_ascii=False)+'。\n\n'
        '当前状态分类：'+json.dumps(status.get('statuses',{}),ensure_ascii=False)+'。\n\n'
        '异常熔断：'+json.dumps(status.get('guards',[]),ensure_ascii=False)+'。\n\n'
        '当前版本允许最多270次调用，加上前序30次最多300次；全局累计最多916次，硬上限1000。'
        'SDK关闭隐式重试，最多44次本地规则返工；报告输入超过20000 Token会告警，失败输出超过20000或任意输出超过40000时停止排队，'
        '本版本报告总Token达到250万也停止排队（在途请求可能有尾量）。所有失败同样记账。\n\n'
        '九条相同生产提示词下的新教师原始答案已复用并重审，因此记录数不等于本版本新调用数；'
        '复用来源和旧规则误判保存在账本中，原始文本不改写。\n\n'
        '此前接口将结构化格式返回成普通文本，并在禁用工具时返回web_search_call；'
        '现已将格式与不联网要求补入真实生产提示词，并继续检查原始响应和异常用量。'
        '服务端是否严格执行仍以每次响应为准，不能仅凭请求参数保证。\n\n'
        '训练候选每条至少15个。西岛/蜈支洲缺渡船证据时从可选池排除并以真实同城点补足；'
        '必去该点的输入隔离。没有用驾车数据代替步行或船班。\n\n'
        '接受表示通过确定性路线、已覆盖的事实/数值检查、严格Schema和3584长度检查，'
        '不等于完整人工语义认证。检查还包括按天范围识别的时长断言，避免把首日120/次日90误作全程矛盾。\n\n'
        '完成后仅导出preview和复核明细；需继续审阅分布、解释、错误拒绝与复杂交通，才发布训练集。'
        '冻结50题不被改写；后续模型比较必须说明当前生产提示词升级，必要时为所有模型增加同协议配对比较。\n\n'
        '参考：[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。\n')
    dest=ROOT/'docs/post_train_large_pool_synthesis_progress.md'
    dest.write_text(text,encoding='utf-8')
    print(json.dumps(dict(records=len(rows),accepted=len(accepted),experiment_calls=starts,global_calls=global_calls,report=str(dest))))


if __name__=='__main__':main()
