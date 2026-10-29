"""Create one full-BF16 evaluation arm at a time, then remove it after evaluation."""
from __future__ import annotations
import argparse,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_online_holdout_matrix import HOST,REMOTE,PY,BASE,V2,SAVES

OUT=REMOTE+'/models/online-holdout-rolling-20261027'
MERGE=REMOTE+'/scripts/post_train_train_entry.py'
SFT={'A':SAVES+'TravelMind-count-ab-20261025-A','B':SAVES+'TravelMind-count-ab-total96-20261025-B'}
DPO={name:SAVES+'shared-dpo-full-20261026-'+name for name in ['v2','A','B']}
DPO_PARENT=REMOTE+'/models/shared-dpo-full-20261026-temporary-merged'
TAGS=['base','v2_sft','A_sft','B_sft','v2_dpo','A_dpo','B_dpo']

def ssh(source):
    result=subprocess.run(['ssh',HOST,PY+' -'],input=source,text=True,capture_output=True,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr[-3000:])
    return result.stdout.strip()

def verify_adapter(path,expected_base):
    info=json.loads(ssh(f"from pathlib import Path;print(Path({(path+'/adapter_config.json')!r}).read_text())"))
    if info.get('base_model_name_or_path')!=expected_base:
        raise RuntimeError(f"Adapter parent mismatch for {path}: {info.get('base_model_name_or_path')}")
    evidence=json.loads(ssh(f"import hashlib,json;from pathlib import Path;p=Path({(path+'/adapter_model.safetensors')!r});print(json.dumps(dict(exists=p.is_file(),sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None)))"))
    if not evidence['exists']:raise RuntimeError('Missing adapter weights: '+path)
    return dict(adapter_path=path,expected_base=expected_base,adapter_sha256=evidence['sha256'])

def all_bindings():
    evidence={'A_sft':verify_adapter(SFT['A'],V2),'B_sft':verify_adapter(SFT['B'],BASE),
              'v2_dpo':verify_adapter(DPO['v2'],V2)}
    for arm in ['A','B']:
        sft_base=V2 if arm=='A' else BASE
        evidence[arm+'_dpo_sft']=verify_adapter(SFT[arm],sft_base)
        evidence[arm+'_dpo']=verify_adapter(DPO[arm],DPO_PARENT)
    return evidence

def merge(tag,base,adapter,output):
    cmd=shlex.join([PY,MERGE,'merge','--base',base,'--adapter',adapter,'--output',output])
    result=subprocess.run(['ssh',HOST,cmd],text=True,capture_output=True,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'{tag} merge failed: {result.stderr[-3000:]}')
    return result.stdout[-1000:]

def checkpoint_bytes(paths):
    return int(ssh(f"""
from pathlib import Path
paths={paths!r};total=0
for name in paths:
 p=Path(name)
 if p.exists():total+=sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
print(total)
"""))

def assert_rolling_peak(paths):
    used=checkpoint_bytes(paths)
    if used>40_000_000_000:
        raise RuntimeError(f'Rolling full-checkpoint peak exceeds 40GB: {used} bytes')
    return used

def guard_merge(source,outputs):
    sizes=json.loads(ssh(f"""
import json,shutil
from pathlib import Path
p=Path({source!r});size=sum(x.stat().st_size for x in p.glob('*.safetensors'))
print(json.dumps(dict(weight_bytes=size,free_bytes=shutil.disk_usage('/home/jiale').free)))
"""))
    projected=sizes['weight_bytes']*outputs
    if not sizes['weight_bytes']:raise RuntimeError('No source safetensors found: '+source)
    if projected>40_000_000_000:raise RuntimeError(f'Predicted rolling checkpoint peak exceeds 40GB: {projected}')
    if sizes['free_bytes']<projected+8_000_000_000:
        raise RuntimeError(f'Insufficient disk headroom: need {projected+8_000_000_000} bytes, have {sizes["free_bytes"]}')
    return dict(source_weight_bytes=sizes['weight_bytes'],predicted_generated_peak_bytes=projected,
                free_before_bytes=sizes['free_bytes'])

def prepare(tag,evidence):
    if tag not in TAGS:raise ValueError(tag)
    target=OUT+'/'+tag
    state=json.loads(ssh(f"""
import json,os,shutil
p={target!r};free=shutil.disk_usage('/home/jiale').free
print(json.dumps(dict(exists=os.path.exists(p),free=free)))
"""))
    if state['exists']:raise RuntimeError(f'Rolling output already exists; inspect before cleanup: {target}')
    # A 7B BF16 merge writes ~15GB. Keep >32GB free to cover parent+child peak and runtime files.
    if state['free']<32_000_000_000:raise RuntimeError('Rolling merge requires at least 32GB free before creation')
    ssh(f"from pathlib import Path;Path({OUT!r}).mkdir(parents=True,exist_ok=True)")
    parent=OUT+'/parent-'+tag
    lineage={}
    if tag in ['base','v2_sft']:
        path=BASE if tag=='base' else V2
    elif tag in ['A_sft','B_sft']:
        arm=tag[0];base=V2 if arm=='A' else BASE
        lineage=dict(sft_base=base,sft_adapter=SFT[arm],evidence=evidence[tag])
        lineage['disk_guard']=guard_merge(base,1)
        merge(tag,base,SFT[arm],target);assert_rolling_peak([target]);path=target
    elif tag=='v2_dpo':
        lineage=dict(sft_base=V2,dpo_adapter=DPO['v2'],evidence=evidence[tag])
        lineage['disk_guard']=guard_merge(V2,1)
        merge(tag,V2,DPO['v2'],target);assert_rolling_peak([target]);path=target
    else:
        arm=tag[0];base=V2 if arm=='A' else BASE
        sft_parent=DPO_PARENT
        lineage=dict(sft_base=base,sft_adapter=SFT[arm],dpo_adapter=DPO[arm],
                     sft_evidence=evidence[arm+'_dpo_sft'],dpo_evidence=evidence[arm+'_dpo'])
        if json.loads(ssh(f"import os,json;print(json.dumps(os.path.exists({sft_parent!r})))")):
            raise RuntimeError('Training parent path already exists; refusing to overwrite: '+sft_parent)
        lineage['disk_guard']=guard_merge(base,2)
        merge(tag+'-sft-parent',base,SFT[arm],sft_parent)
        try:
            assert_rolling_peak([sft_parent])
            guard_merge(sft_parent,2)
            merge(tag+'-dpo',sft_parent,DPO[arm],target)
            assert_rolling_peak([sft_parent,target])
        except Exception:
            cleanup_paths([sft_parent],allow_training_parent=True)
            raise
        cleanup_paths([sft_parent],allow_training_parent=True);path=target
    check=json.loads(ssh(f"""
import hashlib,json
from pathlib import Path
p=Path({path!r});files=sorted(p.glob('*.safetensors'))
if not files:raise RuntimeError('Missing BF16 safetensor weights')
print(json.dumps(dict(path=str(p),bytes=sum(x.stat().st_size for x in files),
 shards=[dict(name=x.name,bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in files],
 config_sha256=hashlib.sha256((p/'config.json').read_bytes()).hexdigest())))
"""))
    return dict(tag=tag,path=path,ephemeral=tag not in ['base','v2_sft'],lineage=lineage,weights=check)

def cleanup_paths(paths,allow_training_parent=False):
    for value in paths:
        ssh(f"""
from pathlib import Path
import shutil
root=Path({OUT!r}).resolve();p=Path({value!r}).resolve()
training_parent=Path({DPO_PARENT!r}).resolve()
if not (p.parent==root and p.name=={Path(value).name!r}) and not ({allow_training_parent!r} and p==training_parent):
 raise RuntimeError('Refuse to remove a path outside the exact rolling output')
if p.exists():shutil.rmtree(p)
""")

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('preflight');prep=sub.add_parser('prepare');prep.add_argument('--tag',choices=TAGS,required=True)
    clean=sub.add_parser('cleanup');clean.add_argument('--tag',choices=TAGS,required=True)
    args=ap.parse_args();evidence=all_bindings()
    if args.action=='preflight':
        state=json.loads(ssh("import json,shutil;print(json.dumps(dict(free=shutil.disk_usage('/home/jiale').free)))"))
        if state['free']<32_000_000_000:raise RuntimeError('Need at least 32GB free before a rolling merge')
        print(json.dumps(dict(free_bytes=state['free'],adapter_evidence=evidence),ensure_ascii=False));return
    if args.action=='prepare':
        result=prepare(args.tag,evidence);print(json.dumps(result,ensure_ascii=False));return
    target=OUT+'/'+args.tag
    cleanup_paths([target,OUT+'/parent-'+args.tag,OUT+'/parent-'+args.tag[0]])
    print(json.dumps(dict(cleaned=args.tag)))

if __name__=='__main__':main()
