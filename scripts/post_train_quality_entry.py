"""Instrument actual training input/supervised tokens without changing loss."""
import json
import os
from pathlib import Path
import time


def main():
    import transformers
    old=transformers.Trainer.training_step
    output=Path(os.environ['TRAVELMIND_TOKEN_LEDGER'])
    output.parent.mkdir(parents=True,exist_ok=True)
    def counted(self,model,inputs,*args,**kwargs):
        # Count once per microbatch, before gradient-checkpoint recomputation.
        row=dict(at=time.time(),optimizer_step_before=int(self.state.global_step),
            samples=int(inputs['input_ids'].shape[0]),
            nonpadding_tokens=int(inputs['attention_mask'].sum().item()),
            supervised_tokens=int((inputs['labels']!=-100).sum().item()))
        value=old(self,model,inputs,*args,**kwargs)
        with output.open('a') as f:
            f.write(json.dumps(row)+'\n')
        return value
    transformers.Trainer.training_step=counted
    from post_train_train_entry import main as train
    train()


if __name__=='__main__':
    main()
