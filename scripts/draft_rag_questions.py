"""Draft annotation candidates. Outputs are NOT human gold labels."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.rag_retrieval import load_chunks


def main():
    chunks = load_chunks()
    cases = []
    # Multiple paraphrases of one evidence group stay within a single split.
    groups = sorted({r["evidence_group"] for r in chunks})
    split = {g: "dev" if i % 2 == 0 else "test" for i, g in enumerate(groups)}
    for i, row in enumerate(chunks):
        relevant = [{"chunk_id": x["chunk_id"], "relevance": 2, "evidence_group": x["evidence_group"]} for x in chunks if x["evidence_group"] == row["evidence_group"]]
        for wording in ("具体要求是什么", "出行前应该注意什么"):
            cases.append({"id": f"draft-{i:03d}-{len(cases)}", "query": row["city"] + " " + row["topic"] + "，" + wording,
                          "type": "draft_answerable", "intent_group": row["evidence_group"], "split": split[row["evidence_group"]],
                          "relevant_chunks": relevant, "reference_facts": [row["text"]], "annotation_status": "pending_review",
                          "annotator": None, "drafted_by": "deterministic_template", "annotation_version": "draft-v1",
                          "change_log": ["Initial machine draft; human review required"]})
    missing = ["上海博物馆今天剩余预约名额", "南京博物院明天停车位库存", "重庆轻轨现在实时延误", "丽江玉龙雪山今天索道余票", "三亚明天逐小时降雨概率", "景德镇今天陶溪川集市实时摊位", "上海博物馆今日轮椅库存", "南京总统府现在排队人数", "重庆洪崖洞今天临时封闭入口", "丽江古城今日酒店空房", "三亚蜈支洲岛当前船票库存", "景德镇御窑博物馆今晚临时闭馆", "上海博物馆今天优惠票价", "南京市区当前出租车报价", "重庆景区今天实时拥挤度"]
    outside = ["Kubernetes 网络策略如何配置", "Python GIL 的实现", "MySQL 事务隔离级别", "Redis 主从复制原理", "Go goroutine 调度机制", "Java 虚拟机垃圾回收", "股票明天一定涨吗", "期货保证金计算", "蛋白质结构预测算法", "如何证明黎曼猜想", "GPU CUDA 内存布局", "TLS 握手流程", "操作系统页表结构", "编译器寄存器分配", "数字电路触发器原理"]
    for i in range(15):
        for kind, query in (("in_corpus_missing", missing[i] + "是什么？"), ("out_of_corpus", outside[i] + "？")):
            cases.append({"id": f"draft-{kind}-{i}", "query": query, "type": kind,
                          "intent_group": f"{kind}-{i}", "split": "dev" if i % 2 == 0 else "test", "relevant_chunks": [],
                          "reference_facts": [], "annotation_status": "pending_review", "annotator": None,
                          "annotation_version": "draft-v1", "change_log": ["Machine draft, pending human review"]})
    path = ROOT / "evaluation" / "rag_benchmark_v1_draft.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cases)} draft candidates; not 120 approved questions, not a gold set")


if __name__ == "__main__":
    main()
