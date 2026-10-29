"""CPU-only exact additive LoRA composition for seven arms on ONE base server.

A/DPO adapters were trained on merged SFT parents. They must NOT be mounted
alone on raw Qwen. For ordinary linear LoRA, concatenate A and scaled B so
sum_i(scale_i B_i A_i) == B_concat A_concat, without a lossy SVD.
"""
import argparse
import hashlib
import json
from pathlib import Path


def compose(sources, output, base):
    import torch
    from safetensors.torch import load_file, save_file
    configs=[json.loads((p/'adapter_config.json').read_text()) for p in sources]
    for c in configs:
        if c.get('use_dora') or c.get('use_rslora') or c.get('modules_to_save') or c.get('bias','none')!='none' or c.get('rank_pattern') or c.get('alpha_pattern'):
            raise RuntimeError('Unsupported non-additive/patterned adapter configuration')
    weights=[load_file(str(p/'adapter_model.safetensors'),device='cpu') for p in sources]
    if any(set(w)!=set(weights[0]) for w in weights):raise RuntimeError('Adapter module keys differ')
    out={};rank=sum(c['r'] for c in configs);max_error=0.
    generator=torch.Generator().manual_seed(271027)
    for key in weights[0]:
        if '.lora_A.' not in key:continue
        bkey=key.replace('.lora_A.','.lora_B.')
        aa=[w[key].float() for w in weights]
        bb=[w[bkey].float()*(c['lora_alpha']/c['r']) for w,c in zip(weights,configs)]
        a=torch.cat(aa,dim=0);b=torch.cat(bb,dim=1)
        x=torch.randn(a.shape[1],3,generator=generator)
        expected=sum(bi@(ai@x) for ai,bi in zip(aa,bb))
        actual=b@(a@x)
        error=(actual-expected).abs().max().item()
        if not torch.allclose(actual,expected,rtol=2e-4,atol=2e-4):raise RuntimeError('Composition check failed: '+key)
        max_error=max(max_error,error)
        out[key]=a.contiguous();out[bkey]=b.contiguous()
    if len(out)!=len(weights[0]):raise RuntimeError('Unexpected non-linear/embedding adapter keys')
    output.mkdir(parents=True,exist_ok=True)
    cfg=dict(configs[0],r=rank,lora_alpha=rank,base_model_name_or_path=base,inference_mode=True,lora_dropout=0.)
    (output/'adapter_config.json').write_text(json.dumps(cfg,indent=2))
    save_file(out,str(output/'adapter_model.safetensors'))
    return dict(rank=rank,max_projection_error=max_error,sources=[dict(path=str(p),
        config_sha256=hashlib.sha256((p/'adapter_config.json').read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256((p/'adapter_model.safetensors').read_bytes()).hexdigest()) for p in sources],
        effective_weights_sha256=hashlib.sha256((output/'adapter_model.safetensors').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/jiale/TravelMind-posttrain'))
    p.add_argument('--base',default='/home/jiale/models/Qwen2.5-7B-Instruct');args=p.parse_args()
    saves=args.root/'saves';out=args.root/'models/online-holdout-adapters-20261027'
    v=saves/'TravelMind-Planner-v2-current-20261025';a=saves/'TravelMind-count-ab-20261025-A';b=saves/'TravelMind-count-ab-total96-20261025-B'
    chains={'v2_sft':[v],'A_sft':[v,a],'B_sft':[b],
            'v2_dpo':[v,saves/'shared-dpo-full-20261026-v2'],
            'A_dpo':[v,a,saves/'shared-dpo-full-20261026-A'],
            'B_dpo':[b,saves/'shared-dpo-full-20261026-B']}
    manifest=dict(base=args.base,composition='exact concatenation of additive LoRA deltas, no SVD, no model generation',variants={})
    for tag,chain in chains.items():
        dest=out/tag
        manifest['variants'][tag]=dict(path=str(dest),**compose(chain,dest,args.base))
        print(json.dumps(dict(prepared=tag,rank=manifest['variants'][tag]['rank'])),flush=True)
    (out/'binding.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
