"""Memory-bounded log probabilities for pinned LLaMA-Factory sigmoid DPO.

Preserves the SFT reference and summed completion log probabilities. Checkpointed
FP32 vocabulary reductions avoid retaining a full FP32 log-softmax for backward.
"""


def completion_logps(logits, labels, chunk_size=64):
    import torch
    from torch.utils.checkpoint import checkpoint

    def reduce_chunk(values, targets):
        values = values.float()
        mask = targets.ne(-100)
        targets = targets.masked_fill(~mask, 0)
        selected = values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return ((selected - values.logsumexp(-1)) * mask).sum(-1)

    labels = labels[:, 1:]
    total = logits.new_zeros(logits.shape[0], dtype=torch.float32)
    for start in range(0, labels.shape[1], chunk_size):
        values = logits[:, start:min(start + chunk_size, labels.shape[1])]
        targets = labels[:, start:start + chunk_size]
        if torch.is_grad_enabled() and values.requires_grad:
            total = total + checkpoint(reduce_chunk, values, targets, use_reentrant=False)
        else:
            total = total + reduce_chunk(values, targets)
    return total, labels.ne(-100).sum(-1)


def low_memory_forward(self, model, batch, is_ref_model=False):
    import torch
    if self.loss_type != 'sigmoid' or self.ld_alpha is not None:
        raise ValueError('Low-memory adapter supports sigmoid DPO without LD weighting only')
    # DPO computes its own loss; passing labels asks the model to also compute
    # an unused full-vocabulary causal cross entropy.
    from torch.utils.checkpoint import checkpoint
    keys = [key for key in batch if key != 'labels']
    def score(labels, *values):
        logits = model(**dict(zip(keys, values)), return_dict=True, use_cache=False).logits
        logps, lengths = completion_logps(logits, labels)
        with torch.no_grad():
            means = logits.mean(dim=(-1, -2), dtype=torch.float32)
        return logps, lengths, means

    # Score each sequence independently and checkpoint the complete scoring
    # path, including the LM head. Only small scores survive ref inference.
    results = []
    for index in range(batch['input_ids'].shape[0]):
        args = (batch['labels'][index:index+1], *(batch[key][index:index+1] for key in keys))
        if torch.is_grad_enabled():
            results.append(checkpoint(score, *args, use_reentrant=False))
        else:
            results.append(score(*args))
    logps, lengths, means = (torch.cat(items) for items in zip(*results))
    count = batch['input_ids'].shape[0] // 2
    chosen, rejected = logps.split(count)
    # Upstream only uses these for mean-logit logging. Keep detached scalars,
    # rather than retaining both vocabulary-sized tensors during ref forward.
    chosen_mean, rejected_mean = means.split(count)
    return chosen, rejected, chosen_mean, rejected_mean, chosen / lengths[:count]


def install():
    import importlib.metadata
    import torch
    from llamafactory.train.dpo.trainer import CustomDPOTrainer

    if importlib.metadata.version('llamafactory') != '0.9.3':
        raise RuntimeError('Low-memory DPO adapter requires reviewed LLaMA-Factory 0.9.3')

    CustomDPOTrainer.concatenated_forward = low_memory_forward
    print('Enabled checkpointed FP32 log-prob reductions; sigmoid DPO reference preserved.', flush=True)
