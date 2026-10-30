"""Preregister, run and audit the no-pool Base protocol. No external LLM calls."""
from __future__ import annotations
import argparse, copy, hashlib, json, random, re, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data/post_train/online_holdout50_20261027_v2"
DEFAULT_OUT = ROOT / "data/post_train/no_pool_base_20261030"
HISTORY = SUITE / "controlled-7arm-localrules-20261028"
SEEDS = [1001, 1002, 1003]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False)+"\n"); f.flush()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_pool(messages):
    out = copy.deepcopy(messages)
    replacements = {
        "景点只能来自景点候选池，餐馆只能来自餐馆候选池，禁止编造名称。": "餐馆只能来自餐馆候选池，禁止编造名称。",
        "知识仅补充候选景点的展陈内容、活动方式或产品组成，不得据此新增候选池外景点。": "",
        "候选不足时只用已有景点并说明缺口，不得编造景点。": "",
        "候选不足时宁可少排。": "",
        "只从给定候选中选择。": "",
        "请用现有候选完成，不因候选少就全部拒绝，也不要重复填充。": "",
        "不要为了凑满候选而加点。": "不要为了凑数而加点。",
        "目前只检索到3个候选": "",
        "候选数量充足，但": "",
        "博物馆类候选": "博物馆类景点", "公园类候选": "公园类景点", "纪念馆类候选": "纪念馆类景点",
        "如果候选有明确室内证据则优先室内，否则仍接受现有候选，但请提醒携带雨具、复核天气，不能把未知说成已核实室内": "如果有明确室内证据则优先室内，否则请提醒携带雨具、复核天气，不能把未知说成已核实室内",
        "检索事实只用于比较已核验候选的兴趣匹配，不得覆盖开放时间、道路通勤等硬约束。": "检索事实不得覆盖开放时间、道路通勤等硬约束。",
    }
    for m in out:
        t=m["content"]
        t=re.sub(r"景点候选池：.*?(?=候选覆盖状态：)", "", t, flags=re.S)
        t=re.sub(r"候选覆盖状态：[^\n]*\n?", "", t)
        for old,new in replacements.items(): t=t.replace(old,new)
        if re.search(r"ReAct|补查动作|查询动作",t,re.I):
            raise ValueError("Unreviewed tool instruction in prompt")
        m["content"]=t
    return out


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/"generations.jsonl").exists(): raise ValueError("Existing results: never overwrite a running experiment")
    cases=read(SUITE/"cases.json")
    old=[]
    for tag in ["base", "C_dpo"]:
        with (HISTORY/tag/"results.jsonl").open(encoding="utf-8") as f:
            old.extend(dict(json.loads(line), arm=tag) for line in f if json.loads(line)["mode"]=="planner-only")
    source=next(r for r in old if r["arm"]=="base")
    schema=copy.deepcopy(source["calls"][0]["request"]["response_format"])
    prop=schema["json_schema"]["schema"]["$defs"]["SpotPlan"]["properties"]["name"]
    assert prop["type"]=="string" and "enum" not in prop
    prop["description"]="景点名称，自由文本"
    requests=[]
    for c in cases:
        messages=strip_pool(c["messages"])
        for m in messages:
            if "景点候选池" in m["content"] or "只从给定候选" in m["content"] or "现有候选" in m["content"]:
                raise ValueError("Residual pool-only instruction: "+c["id"])
        for seed in SEEDS:
            requests.append(dict(id=c["id"],city=c["city"],expected=c["expected"],rubric=c["rubric"],
                payload=dict(model="Qwen2.5-7B-Instruct",messages=messages,temperature=.3,top_p=.95,seed=seed,
                             max_tokens=2048,stream=False,response_format=schema)))
    random.Random(271027).shuffle(requests)
    dump(out/"requests.json",requests)
    # Immutable first-round historical outputs, rescored by the same verifier later.
    baseline=[]
    for r in old:
        baseline.append({"arm":r["arm"],"id":r["id"],"seed":r["seed"],"parse_success":r["delivery"],
            "plan":r.get("plan"),"entity_total":r["entity_total"],"entity_outside":r["entity_outside"]})
    dump(out/"historical_first_round.json",baseline)
    audit={}
    ids={c["id"] for c in cases if c["expected"]=="feasible"}
    for tag in ["base","C_dpo"]:
        audit[tag]={}
        for group,rows in [("all150",[r for r in baseline if r["arm"]==tag]),("feasible135",[r for r in baseline if r["arm"]==tag and r["id"] in ids])]:
            n=sum(r["entity_total"] for r in rows); k=sum(r["entity_outside"] for r in rows)
            audit[tag][group]=dict(runs=len(rows),entity_total=n,entity_outside=k,outside_rate_pct=100*k/n if n else None)
    dump(out/"baseline_denominators.json",audit)
    p=dict(suite_sha256=sha(SUITE/"cases.json"),requests_sha256=sha(out/"requests.json"),script_sha256=sha(__file__),
        cases=50,feasible_runs=135,infeasible_runs=15,seeds=SEEDS,temperature=.3,top_p=.95,max_tokens=2048,
        concurrency=1,warmup=5,prefix_cache=False,schema_changes=["$defs.SpotPlan.properties.name.description only; already free text, no IDs/enums"],
        retry="Only JSON/Schema parsing failure: one identical payload retry; separate ledger; never replace first attempt",
        verification="AMap prefetch then offline. API failures/polygon uncertainty stay pending; never label as d.",
        metrics="b+c+d / all entity occurrences in first-successfully-parsed feasible runs; empty/failed outputs separately reported",
        caveat="Historical pool-outside rate is NOT geographic unverifiability; same verifier regrades archived outputs without generation",
        human_review="seeded stratified sample >=12 per occupied bucket, all d; human labels remain pending until reviewed")
    dump(out/"protocol.json",p)
    dump(out/"cases_metadata.json",[{k:c[k] for k in ["id","city","expected","rubric","soft_rubric"]} for c in cases])
    dump(out/"pool_evidence.json",[{"city":c["city"],"poi":poi} for c in cases for poi in c["state"]["pois"]])
    print(json.dumps(dict(status="prepared",requests=len(requests),baseline_denominators=audit),ensure_ascii=False))


def call(url,payload):
    req=urllib.request.Request(url.rstrip('/')+"/chat/completions",data=json.dumps(payload,ensure_ascii=False).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer EMPTY"})
    started=time.monotonic()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=90) as response: data=json.loads(response.read())
    return data,time.monotonic()-started


def parse_response(data,schema):
    import jsonschema
    text=data["choices"][0]["message"]["content"]
    obj=json.loads(text); jsonschema.validate(obj,schema)
    if data["choices"][0].get("finish_reason")!="stop": raise ValueError("non-stop finish reason")
    return obj


def generate(args):
    out=args.output; requests=read(out/"requests.json"); p=read(out/"protocol.json")
    assert sha(out/"requests.json")==p["requests_sha256"]
    assert sha(__file__)==p["script_sha256"],"Generator differs from preregistered code"
    if not (out/"warmup.json").exists():
        warm=[]
        for r in requests[:5]:
            data,elapsed=call(args.url,r["payload"]);warm.append(dict(usage=data.get("usage"),elapsed=elapsed))
        dump(out/"warmup.json",dict(discarded=True,calls=warm))
    path=out/"generations.jsonl"
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done={(r["id"],r["seed"]) for r in rows}
    for r in requests:
        key=(r["id"],r["payload"]["seed"])
        if key in done: continue
        entry={k:r[k] for k in ["id","city","expected"]};entry["seed"]=key[1];entry["attempts"]=[]
        for attempt in range(2):
            data,elapsed=call(args.url,r["payload"]) # transport failures stop; no silent resampling
            a=dict(attempt=attempt,response=data,elapsed_s=elapsed,parse_success=False)
            try:
                a["plan"]=parse_response(data,r["payload"]["response_format"]["json_schema"]["schema"]); a["parse_success"]=True
            except (ValueError,KeyError,TypeError) as exc: a["parse_error"]=str(exc)
            except Exception as exc:
                if type(exc).__name__ not in {"ValidationError"}: raise
                a["parse_error"]=str(exc)
            append(out/"attempts.jsonl",dict(id=r["id"],seed=key[1],**a));entry["attempts"].append(a)
            if a["parse_success"]: break
        entry["first_parse_success"]=entry["attempts"][0]["parse_success"]
        append(path,entry);rows.append(entry)
        dump(out/"status.json",dict(completed=len(rows),target=150,last_id=r["id"],first_parse_failures=sum(not x["first_parse_success"] for x in rows),updated_at=time.time()))
        print(f"completed {len(rows)}/150",flush=True)
    dump(out/"generation_summary.json",dict(total=len(rows),first_parse_failures=sum(not r["first_parse_success"] for r in rows),
        retry_recovered=sum(len(r["attempts"])==2 and r["attempts"][1]["parse_success"] for r in rows),status="complete"))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dry-run",action="store_true");ap.add_argument("--prepare",action="store_true");ap.add_argument("--run",action="store_true")
    ap.add_argument("--url",default="http://127.0.0.1:8000/v1");ap.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    if args.prepare or args.dry_run: prepare(args.output)
    elif args.run: generate(args)
    else: ap.error("Choose --prepare/--dry-run or --run")
if __name__=="__main__": main()
