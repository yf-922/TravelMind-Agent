"""Read-only allowlisted remote training evidence, never environment secrets."""
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

ROOT=Path('/home/jiale/TravelMind-posttrain')
RUNS={'v2_sft':'TravelMind-Planner-v2-current-20261025',
      'A_sft':'TravelMind-count-ab-20261025-A',
      'B_sft':'TravelMind-count-ab-total96-20261025-B',
      'B_old48':'TravelMind-count-ab-20261025-B',
      **{a+'_dpo':'shared-dpo-full-20261026-'+a for a in ['v2','A','B']}}
FIELDS=['learning_rate','num_train_epochs','max_steps','per_device_train_batch_size',
        'gradient_accumulation_steps','lr_scheduler_type','warmup_ratio','warmup_steps',
        'bf16','fp16','optim','weight_decay','max_grad_norm','seed','data_seed',
        'gradient_checkpointing','dataloader_drop_last','world_size','save_strategy',
        'eval_strategy','load_best_model_at_end','metric_for_best_model','greater_is_better']

def main():
    import torch
    result={'runs':{},'versions':{},'defaults':{},'sources':{}}
    for name in ['torch','transformers','peft','llamafactory','trl','accelerate','bitsandbytes','vllm']:
        try: result['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['versions'][name]='not_installed_in_this_environment'
    for tag,folder in RUNS.items():
        path=ROOT/'saves'/folder
        row={'path':str(path),'exists':path.exists()}
        for name in ['trainer_state.json','train_results.json','eval_results.json','adapter_config.json','all_results.json']:
            p=path/name
            if p.exists(): row[name]=json.loads(p.read_text())
        p=path/'training_args.bin'
        if p.exists():
            args=torch.load(p,map_location='cpu',weights_only=False)
            row['resolved_training_args']={key:str(getattr(args,key)) if key in ['lr_scheduler_type','optim','save_strategy','eval_strategy'] else getattr(args,key,None) for key in FIELDS}
        else: row['resolved_training_args']='missing'
        result['runs'][tag]=row
    for arm in ['v2','A','B']:
        p=ROOT/f'configs/post_train/shared_{arm}_planner_dpo_full.yaml'
        result['runs'][arm+'_dpo']['actual_yaml']=p.read_text() if p.exists() else None
    base=Path('/home/jiale/models/Qwen2.5-7B-Instruct/config.json')
    result['base_config']=json.loads(base.read_text())
    from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments
    for cls in [DataArguments,FinetuningArguments,ModelArguments]:
        result['defaults'][cls.__name__]={}
        for f in dataclasses.fields(cls):
            if f.name in ['train_on_prompt','mask_history','packing','neat_packing','lora_dropout','pref_ftx','pref_beta','pref_loss','use_ref_model','quantization_type','double_quantization','quantization_bit','quantization_method']:
                value=f.default if f.default is not dataclasses.MISSING else 'default_factory_or_required'
                result['defaults'][cls.__name__][f.name]=value
    from llamafactory.train.dpo.trainer import CustomDPOTrainer
    result['sources']['reference']=inspect.getsource(CustomDPOTrainer.compute_reference_log_probs)
    result['sources']['dpo_init']=inspect.getsource(CustomDPOTrainer.__init__)
    from llamafactory.train.dpo import workflow
    result['sources']['dpo_workflow']=inspect.getsource(workflow.run_dpo)
    print(json.dumps(result,default=str))

if __name__=='__main__': main()
