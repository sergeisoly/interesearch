import pytest
import torch

from interesearch.assets import MODULES
from interesearch.edits import applied_update, assert_frozen, capture_base, logits_of, snapshot_params
from interesearch.model import LlamaSimpleMLP, LlamaSimpleMLPConfig
from interesearch.training import DenseUpdate, merged_forward, window_ce

V = 64


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = LlamaSimpleMLPConfig(block_size=32, vocab_size=V, n_layer=4, n_head=2, n_embd=32, n_intermediate=64,
                               n_ctx=32, n_key_value_heads=2, flash_attention=False)
    return LlamaSimpleMLP(cfg).eval().requires_grad_(False)


def tokens(seed, n=3, t=17):
    return torch.randint(0, V, (n, t), generator=torch.Generator().manual_seed(seed))


def update(base, modules, seed, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    return {m: scale * torch.randn(base[m].shape, generator=g) for m in modules}


def weights_equal(model, ref):
    return all(torch.equal(model.get_submodule(m).weight, w) for m, w in ref.items())


def test_edit_is_exact_and_restored_after_an_exception(model):
    base, x = capture_base(model, MODULES), tokens(0)
    ref = logits_of(model(x))
    Q = update(base, MODULES[:3], seed=1)
    with applied_update(model, base, None):
        assert torch.equal(logits_of(model(x)), ref)
    with pytest.raises(ZeroDivisionError):
        with applied_update(model, base, Q):
            assert weights_equal(model, {m: base[m] + Q[m] for m in Q})
            assert weights_equal(model, {m: base[m] for m in MODULES[3:]})
            1 / 0
    assert weights_equal(model, base)
    assert torch.equal(logits_of(model(x)), ref)
    with applied_update(model, base, Q):  # the failed edit left nothing active
        pass


def test_reference_block_runs_base_weights_and_writes_the_edit_back(model):
    base, x = capture_base(model, MODULES), tokens(0)
    ref = logits_of(model(x))
    Q = update(base, MODULES[4:6], seed=2)
    with applied_update(model, base, Q) as edit:
        edited = logits_of(model(x))
        with edit.reference():
            assert torch.equal(logits_of(model(x)), ref)
            with pytest.raises(RuntimeError):
                with edit.reference():
                    pass
        assert weights_equal(model, {m: base[m] + Q[m] for m in Q})
        assert torch.equal(logits_of(model(x)), edited)
        with pytest.raises(ZeroDivisionError):
            with edit.reference():
                1 / 0
        assert torch.equal(logits_of(model(x)), edited)
        layer = model.get_submodule(MODULES[4])
        weight = layer.weight
        with pytest.raises(RuntimeError):  # writing the edit back fails: the block is still left
            with edit.reference():
                layer.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        layer.weight = weight
        with edit.reference():
            assert torch.equal(logits_of(model(x)), ref)
    assert weights_equal(model, base)
    with pytest.raises(RuntimeError):
        with edit.reference():
            pass


def test_nested_edit_on_the_same_model_is_refused(model):
    base, other_base = capture_base(model, MODULES), capture_base(model, MODULES[:2])
    Q = update(base, MODULES[:2], seed=3)
    with applied_update(model, base, Q) as edit:
        for b, u in ((base, None), (other_base, None), (other_base, Q)):
            with pytest.raises(RuntimeError):
                with applied_update(model, b, u):
                    pass
        with edit.reference():
            with pytest.raises(RuntimeError):
                with applied_update(model, other_base, None):
                    pass
        with pytest.raises(RuntimeError):  # would capture the edited weights as W_0
            capture_base(model, MODULES)
        assert weights_equal(model, {m: base[m] + Q[m] for m in Q})
    assert weights_equal(model, base)


def test_invalid_update_raises_before_any_write(model):
    base = capture_base(model, MODULES)
    m = MODULES[0]
    bad = [
        (KeyError, {"h.9.attn.q_proj": torch.zeros(1)}),
        (ValueError, {m: torch.zeros(base[m].shape[1])}),  # would broadcast over rows
        (ValueError, {m: torch.zeros(base[m].shape).index_fill_(1, torch.tensor([3]), float("nan"))}),
        (ValueError, {m: torch.full(base[m].shape, 1e300, dtype=torch.float64)}),  # inf after the float32 cast
    ]
    for err, u in bad:
        with pytest.raises(err):
            with applied_update(model, base, {MODULES[1]: torch.ones(base[MODULES[1]].shape), **u}):
                pass
        assert weights_equal(model, base)
    row = {**base, m: base[m][:1]}  # a base row would broadcast over the weight
    for u in (None, {m: torch.zeros(base[m][:1].shape)}):
        with pytest.raises(ValueError):
            with applied_update(model, row, u):
                pass
        assert weights_equal(model, base)
    with applied_update(model, base, None):
        pass


def test_matrix_update_starts_at_the_base_model_and_trains_only_its_deltas(model):
    base, x = capture_base(model, MODULES), tokens(4)
    frozen = snapshot_params(model)
    upd = DenseUpdate({m: tuple(base[m].shape) for m in MODULES})
    fwd = merged_forward(model, base, upd)
    assert torch.equal(fwd(x[:, :-1]), logits_of(model(x[:, :-1])))
    opt = torch.optim.Adam(upd.parameters(), lr=1e-3)
    window_ce(fwd(x[:, :-1]), x[:, 1:]).mean().backward()
    opt.step()
    assert all(D.abs().max() > 0 for D in upd.update().values())
    assert_frozen(model, frozen)
    with torch.no_grad():
        model.ln_f.weight.add_(1.0)
    with pytest.raises(AssertionError):
        assert_frozen(model, frozen)
