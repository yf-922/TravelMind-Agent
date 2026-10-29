"""Controlled route benchmark with streaming timing and granular metrics."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, statistics, time, urllib.request
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from app.planning.schemas import TravelRoute
from scripts.build_representative_eval50 import score_case, OUT as SUITE
from scripts.evaluate_representative50 import verify
from scripts.prepare_sft_v2_release import explanation_faults
SEEDS=[42,43,44]; MAX_REWORK=3

def stream_call(url, model, messages, seed, max_tokens=2048, timeout=180):
    payload=dict(model=model,messages=messages,temperature=0.0,seed=seed,max_tokens=max_tokens,stream=True,
        stream_options={"include_usage":True},response_format={"type":"json_schema","json_schema":{"name":"TravelRoute","strict":False,"schema":TravelRoute.model_json_schema()}})
    t0=time.perf_counter(); ttft=None; text=""; usage={}; finish=None
    with urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(payload,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"}),timeout=timeout) as response:
        for raw in response:
            line=raw.decode("utf8","replace").strip()
            if not line.startswith("data:"): continue
            if line[5:].strip()=="[DONE]": continue
            try: item=json.loads(line[5:].strip())
            except json.JSONDecodeError: continue
            if item.get("usage"): usage=item["usage"]
            for choice in item.get("choices") or []:
                finish=choice.get("finish_reason") or finish
                delta=(choice.get("delta") or {}).get("content")
                if delta:
                    if ttft is None: ttft=time.perf_counter()-t0
                    text+=delta
    e2e=time.perf_counter()-t0; completion=int(usage.get("completion_tokens") or max(1,len(text)//2))
    tpot=(e2e-ttft)/(completion-1)*1000 if ttft is not None and completion>1 else None
    return text,dict(prompt_tokens=int(usage.get("prompt_tokens") or 0),completion_tokens=completion,output_chars=len(text),e2e_s=e2e,ttft_s=ttft,tpot_ms=tpot,finish_reason=finish)

def flags(faults):
    f=set(faults)
    return dict(in_pool_pass="unknown_poi" not in f,opening_pass=not bool(f&{"opening_evidence","opening_time_conflict"}),
        timeline_transit_pass=not bool(f&{"commute_time_conflict","road_unverified","user_time_window","user_break_overlap","route_structure"}),
        non_duplicate_pass="duplicate_poi" not in f)

def summarize(rows, mode):
    n=len(rows); pct=lambda k:100*sum(bool(r[k]) for r in rows)/n if n else None
    out=dict(n=n,delivery_rate_pct=pct("delivery"),all_rule_pass_pct=pct("all_rule_pass"),in_pool_faithfulness_pct=pct("in_pool_pass"),
        opening_hours_pass_pct=pct("opening_pass"),timeline_transit_pass_pct=pct("timeline_transit_pass"),non_duplicate_pass_pct=pct("non_duplicate_pass"),
        avg_prompt_tokens=statistics.mean(r["prompt_tokens"] for r in rows),avg_completion_tokens=statistics.mean(r["completion_tokens"] for r in rows),
        avg_output_chars=statistics.mean(r["output_chars"] for r in rows),total_tokens=sum(r["prompt_tokens"]+r["completion_tokens"] for r in rows),
        latency_p50_s=statistics.median(r["e2e_s"] for r in rows),latency_p95_s=sorted(r["e2e_s"] for r in rows)[math.ceil(.95*n)-1],
        tpot_p50_ms=statistics.median([r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]))
    if mode=="rules-rework":
        triggered=sum(r["rounds_used"]>1 for r in rows); recovered=sum(r["all_rule_pass"] and not r["round0_pass"] for r in rows)
        out.update(rework_trigger_rate_pct=100*triggered/n,avg_rework_turns=statistics.mean(max(0,r["rounds_used"]-1) for r in rows),rework_recovery_rate_pct=100*recovered/triggered if triggered else None)
    return out

def run(args):
    verify(); cases=json.loads((SUITE/"cases.json").read_text(encoding="utf8")); maps=json.loads((SUITE/"maps.json").read_text(encoding="utf8"))["routes"]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/"binding.json").write_text(json.dumps(dict(model=args.model,temperature=0.0,max_tokens=args.max_tokens,seeds=SEEDS,cases=50,mode=args.mode,concurrency=args.concurrency,prefix_cache="disabled_by_launcher",runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding="utf8")
    for case in cases[:5]:
        try: stream_call(args.url,args.model,case["messages"],42,args.max_tokens)
        except Exception: pass
    rows=[]; raw=out/"results.jsonl"
    for seed in SEEDS:
        for case in cases:
            rounds=[]; feedback=""; started=time.perf_counter(); final=None
            for turn in range(1 if args.mode=="planner-only" else MAX_REWORK+1):
                messages=copy.deepcopy(case["messages"])
                if feedback: messages[-1]["content"]+="\n\n确定性规则返工反馈："+feedback
                try:
                    text,timing=stream_call(args.url,args.model,messages,seed,args.max_tokens)
                    plan=TravelRoute.model_validate_json(text).model_dump(mode="json"); faults=score_case(plan,case,maps)
                    rounds.append(dict(turn=turn+1,faults=faults,explanation_faults=explanation_faults(plan),timing=timing,output=plan)); final=plan
                    if not faults or args.mode=="planner-only": break
                    feedback="；".join(faults)
                except Exception as exc:
                    rounds.append(dict(turn=turn+1,error=type(exc).__name__,timing=locals().get("timing",{}))); break
            faults=rounds[-1].get("faults",["delivery_failure"]) if rounds else ["delivery_failure"]; fl=flags(faults)
            row=dict(id=case["id"],seed=seed,delivery=final is not None,all_rule_pass=final is not None and not faults,round0_pass=bool(rounds and not rounds[0].get("faults",["delivery_failure"])),rounds_used=len(rounds),**fl,
                prompt_tokens=sum(x.get("timing",{}).get("prompt_tokens",0) for x in rounds),completion_tokens=sum(x.get("timing",{}).get("completion_tokens",0) for x in rounds),output_chars=sum(x.get("timing",{}).get("output_chars",0) for x in rounds),e2e_s=time.perf_counter()-started,tpot_ms=statistics.mean([x["timing"]["tpot_ms"] for x in rounds if x.get("timing",{}).get("tpot_ms") is not None]) if rounds else None,faults=faults,rounds=rounds)
            with raw.open("a",encoding="utf8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
            rows.append(row)
    summary=summarize(rows,args.mode); (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8"); return summary

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--url",default="http://127.0.0.1:8000/v1/chat/completions"); p.add_argument("--model",required=True); p.add_argument("--mode",choices=["planner-only","rules-rework"],required=True); p.add_argument("--output",required=True); p.add_argument("--max-tokens",type=int,default=2048); p.add_argument("--concurrency",type=int,default=1); print(json.dumps(run(p.parse_args()),ensure_ascii=False))
