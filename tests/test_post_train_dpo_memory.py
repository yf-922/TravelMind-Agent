"""Verify the memory optimization preserves completion scores and gradients."""
import pytest

torch = pytest.importorskip('torch')
from scripts.post_train_low_memory_dpo import completion_logps


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_chunked_logps_and_dpo_gradients_match(dtype):
    torch.manual_seed(71)
    logits = torch.randn(4, 17, 83, dtype=dtype, requires_grad=True)
    baseline = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, 83, (4, 17))
    labels[:, :5] = -100
    labels[1, -3:] = -100
    actual, lengths = completion_logps(logits, labels, chunk_size=3)
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    expected = baseline[:, :-1].float().log_softmax(-1).gather(
        -1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    expected = (expected * mask).sum(-1)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(lengths, mask.sum(-1))
    def loss(scores):
        return -torch.nn.functional.logsigmoid(.1 * (scores[:2] - scores[2:])).mean()
    loss(actual).backward()
    loss(expected).backward()
    torch.testing.assert_close(logits.grad, baseline.grad, rtol=.02 if dtype == torch.bfloat16 else 1e-5,
                               atol=2e-4 if dtype == torch.bfloat16 else 1e-7)


def test_sequential_checkpoint_forward_preserves_model_gradients():
    from types import SimpleNamespace
    from scripts.post_train_low_memory_dpo import low_memory_forward
    torch.manual_seed(8)
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(31, 12)
            self.head = torch.nn.Linear(12, 31)
        def forward(self, input_ids, **kwargs):
            assert 'labels' not in kwargs
            return SimpleNamespace(logits=self.head(self.embed(input_ids)))
    import copy
    model = TinyModel()
    baseline = copy.deepcopy(model)
    ids = torch.randint(0, 31, (4, 9))
    labels = ids.clone()
    labels[:, :3] = -100
    batch = {'input_ids': ids, 'labels': labels}
    trainer = SimpleNamespace(loss_type='sigmoid', ld_alpha=None)
    chosen, rejected, _, _, avg = low_memory_forward(trainer, model, batch)
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    expected = baseline(ids).logits[:, :-1].log_softmax(-1).gather(
        -1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    expected = (expected * mask).sum(-1)
    torch.testing.assert_close(torch.cat((chosen, rejected)), expected)
    torch.testing.assert_close(avg, expected[:2] / mask[:2].sum(-1))
    (-torch.nn.functional.logsigmoid(.1 * (chosen-rejected))).mean().backward()
    (-torch.nn.functional.logsigmoid(.1 * (expected[:2]-expected[2:]))).mean().backward()
    for actual, reference in zip(model.parameters(), baseline.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-5, atol=1e-7)
    with torch.no_grad():
        ref = low_memory_forward(trainer, model, batch, is_ref_model=True)
    torch.testing.assert_close(ref[0], chosen.detach())
