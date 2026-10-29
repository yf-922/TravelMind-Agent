"""Run pinned LLaMA-Factory and record PyTorch peak allocation for each stage."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    command = sys.argv.pop(1)
    if command == 'merge':
        p = argparse.ArgumentParser()
        p.add_argument('--base', required=True)
        p.add_argument('--adapter', required=True)
        p.add_argument('--output', required=True)
        a = p.parse_args()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map='cpu', local_files_only=True)
        merged = PeftModel.from_pretrained(base, a.adapter, local_files_only=True).merge_and_unload()
        merged.save_pretrained(a.output, safe_serialization=True, max_shard_size='4GB')
        AutoTokenizer.from_pretrained(a.base, local_files_only=True).save_pretrained(a.output)
        print(json.dumps({'stage': 'merge', 'output': a.output}))
        return
    if command != 'train':
        raise ValueError('expected train or merge')
    import torch
    if any('planner_dpo' in arg for arg in sys.argv):
        from post_train_low_memory_dpo import install
        install()
    from llamafactory.train.tuner import run_exp
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    success = False
    try:
        run_exp()
        success = True
    finally:
        result = {'stage_config': sys.argv[1], 'success': success, 'elapsed_s': time.time()-started,
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
            'peak_reserved_gib': torch.cuda.max_memory_reserved()/1024**3}
        Path('logs').mkdir(exist_ok=True)
        with Path('logs/gpu_training_metrics.jsonl').open('a') as f:
            f.write(json.dumps(result)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
