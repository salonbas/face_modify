import pytest
torch = pytest.importorskip('torch')
from advface.attacks.mi_fgsm import diverse_input


def test_disabled_di_is_identity_and_does_not_consume_rng():
    value = torch.randn(1, 3, 112, 112, requires_grad=True)
    generator = torch.Generator().manual_seed(5)
    state = generator.get_state().clone()
    output, applied = diverse_input(value, probability=0, generator=generator)
    assert output is value and not applied
    assert torch.equal(state, generator.get_state())


def test_di_is_differentiable_shape_preserving_and_seeded():
    value = torch.ones(1, 3, 112, 112, requires_grad=True)
    output, applied = diverse_input(value, probability=1, min_scale=0.5,
                                   generator=torch.Generator().manual_seed(0))
    repeated, _ = diverse_input(value, probability=1, min_scale=0.5,
                               generator=torch.Generator().manual_seed(0))
    assert applied and output.shape == value.shape
    assert torch.equal(output, repeated)
    assert output.min() == 0 and output.max() == 1
    output.sum().backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


@pytest.mark.parametrize('probability,min_scale', [(-0.1, 0.9), (1.1, 0.9), (0.7, 0), (0.7, 1.1)])
def test_di_rejects_invalid_settings(probability, min_scale):
    with pytest.raises(ValueError):
        diverse_input(torch.ones(1, 3, 112, 112), probability=probability, min_scale=min_scale)
