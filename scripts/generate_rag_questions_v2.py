"""Exactly 120 diverse machine proposals, never gold annotations."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.evaluation.rag_protocol import CITIES,validate_corpus

INTENTS=[('预约','怎么预约，需要提前准备什么？'),('开放','几点开放，有哪些时间限制？'),
 ('交通','怎样到达，入口和换乘怎么选？'),('亲子','带孩子出行，有什么注意事项？'),
 ('老人','带老人慢慢游览，有什么安排建议？'),('雨天','遇上下雨，游览时应注意什么？'),
 ('门票','购票和入场有哪些条件？'),('安全','游览有哪些安全提醒？'),
 ('设施','有哪些服务设施和使用要求？'),('路线','不想赶路，游览路线怎么安排？')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=json.loads(a.corpus.read_text(encoding='utf-8'))
    errors=validate_corpus(rows,True)
    if errors:p.error('; '.join(errors))
    cases=[]
    # 15 proposals/city: five two-wording groups and five single requests.
    for ci,city in enumerate(CITIES):
        city_rows=[r for r in rows if r['city']==city]
        for i,(term,question) in enumerate(INTENTS):
            evidence=sorted(city_rows,key=lambda r:(-(term in r['text']),r['chunk_id']))[:2 if i>=8 else 1]
            group=f'{city}-intent-{i}'
            # Rotate split assignment by city; each city contributes 7 or 8.
            split='dev' if (i+ci)%2==0 else 'test'
            for variant in range(2 if i<5 else 1):
                query=f'{city} {evidence[0]["topic"]}，{question}' if not variant else f'我准备去{city}，关于{term}的事想问一下：{question}'
                cases.append({'id':f'v2-{ci}-{i}-{variant}','query':query,'city':city,'type':'answerable',
                    'intent_group':group,'split':split,'challenge':['exact_name','colloquial','multi_evidence'][2 if i>=8 else variant],
                    'relevant_chunks':[{'chunk_id':r['chunk_id'],'relevance':2,'evidence_group':r['evidence_group']} for r in evidence],
                    'reference_facts':[r['text'] for r in evidence], 'annotation_status':'pending_review',
                    'annotator':None,'reviewed_at':None,'annotation_version':'v2-draft','drafted_by':'template_evidence_proposal',
                    'change_log':['Proposed evidence may not answer query; human must revise query, labels and facts.']})
    missing=['今天剩余预约名额','明天实时降雨概率','现在排队分钟数','当前停车空位','今日临时闭馆通知']
    outside=['杭州西湖明天游船时刻','北京故宫入场要求','成都博物馆怎么预约','南京如何办理签证延期','上海工作居住证政策',
        '重庆房贷利率是多少','昆明酒店当前库存','三亚潜水资格证考试报名','景德镇陶瓷出口报关','西安兵马俑公交路线',
        '苏州博物馆预约','青岛海边实时风浪','昆明机场转机行李规则','MySQL事务隔离原理','如何修改Redis过期策略']
    for i in range(15):
        for kind,query in [('in_corpus_missing',CITIES[i%6]+missing[i%5]+'是多少？'),('out_of_corpus',outside[i]+'？')]:
            # 45 positive dev +8 missing +7 outside =60; test inverse.
            dev=(i<8) if kind=='in_corpus_missing' else (i<7)
            cases.append({'id':f'v2-{kind}-{i}','query':query,'type':kind,'intent_group':f'{kind}-{i}',
                'split':'dev' if dev else 'test','relevant_chunks':[],'reference_facts':[],
                'annotation_status':'pending_review','annotator':None,'reviewed_at':None,'annotation_version':'v2-draft',
                'change_log':['Human must verify absence from corpus and classify boundary.']})
    a.out.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('120 proposals; human review required, not measured gold accuracy')


if __name__=='__main__':main()
