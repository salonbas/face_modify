import pytest
torch = pytest.importorskip('torch')
from advface.attacks.mi_fgsm import gaussian_ti_kernel, translation_invariant_gradient


def test_gaussian_kernel_is_normalized_positive_and_symmetric():
    kernel = gaussian_ti_kernel(5, 1.0, channels=3, device='cpu', dtype=torch.float32)
    assert kernel.shape == (3, 1, 5, 5)
    assert (kernel > 0).all()
    assert torch.allclose(kernel.sum(dim=(1, 2, 3)), torch.ones(3))
    assert torch.equal(kernel, kernel.flip(-1))
    assert torch.equal(kernel, kernel.flip(-2))


def test_constant_gradient_is_preserved_including_edges():
    gradient = torch.full((2, 3, 11, 13), -3.0)
    assert torch.allclose(translation_invariant_gradient(gradient), gradient, atol=1e-6)


def test_impulse_is_spread_without_channel_mixing():
    gradient = torch.zeros(1, 3, 13, 13)
    gradient[0, 1, 6, 6] = 1
    result = translation_invariant_gradient(gradient)
    assert result[0, 0].abs().sum() == 0
    assert result[0, 2].abs().sum() == 0
    assert torch.isclose(result.sum(), torch.tensor(1.0))
    assert 0 < result[0, 1, 6, 6] < 1
    assert torch.count_nonzero(result) == 25


def test_smoothing_commutes_with_translation_away_from_boundaries():
    gradient = torch.randn(1, 3, 25, 25)
    shifted = torch.roll(gradient, (1, 2), dims=(-2, -1))
    actual = translation_invariant_gradient(shifted)
    expected = torch.roll(translation_invariant_gradient(gradient), (1, 2), dims=(-2, -1))
    assert torch.allclose(actual[:, :, 5:-5, 5:-5], expected[:, :, 5:-5, 5:-5], atol=1e-6)


def test_size_one_is_exact_identity():
    gradient = torch.randn(1, 3, 13, 13)
    assert translation_invariant_gradient(gradient, kernel_size=1) is gradient


@pytest.mark.parametrize('size,sigma', [(0, 1), (2, 1), (-1, 1), (5, 0), (5, -1), (5, float('nan'))])
def test_invalid_ti_settings_are_rejected(size, sigma):
    with pytest.raises(ValueError):
        gaussian_ti_kernel(size, sigma, channels=3, device='cpu', dtype=torch.float32)
