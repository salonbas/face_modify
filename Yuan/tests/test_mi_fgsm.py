"""Minimal MI-FGSM integration tests; no face models are loaded."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from advface.attacks.mi_fgsm import momentum_update, normalize_gradient_l1
from advface.attacks.registry import AttackConfig, AttackOutput, get_attack_spec, apply_attack, normalize_attack_name
from advface.config import PIXEL_MAX
from advface.evaluation.perturbation import perturbation_metrics, validate_linf


def test_registry_resolves_mi_fgsm():
    spec = get_attack_spec("mi_fgsm")
    assert spec.name == "mi_fgsm"
    assert spec.requires_steps
    assert spec.linf_constraint_domain == "full_image"
    assert normalize_attack_name("mi_fgsm") == "mi_fgsm"


def test_gradient_normalization_is_finite_and_per_image():
    gradient = torch.tensor([[[[3.0, 4.0]]], [[[0.0, 0.0]]]])
    normalized = normalize_gradient_l1(gradient)
    assert torch.isfinite(normalized).all()
    assert normalized[0].abs().sum().item() == pytest.approx(1.0)
    assert torch.equal(normalized[1], torch.zeros_like(normalized[1]))


def test_momentum_accumulates_instead_of_resetting():
    first = torch.tensor([[[[1.0, -1.0]]]])
    second = torch.tensor([[[[-1.0, 1.0]]]])
    accumulated = momentum_update(torch.zeros_like(first), first, 1.0)
    accumulated = momentum_update(accumulated, second, 1.0)
    # With recurrence g2 = g1 + normalized(second), cancellation is exact.
    assert torch.allclose(accumulated, torch.zeros_like(accumulated))
    # A reset-per-step implementation would retain the second gradient.
    assert not torch.allclose(accumulated, normalize_gradient_l1(second))


def test_mi_fgsm_output_metadata_and_linf(monkeypatch):
    original = np.full((2, 2, 3), 100, dtype=np.uint8)
    adversarial = np.full_like(original, 102)

    def fake_run(*args, **kwargs):
        return adversarial

    monkeypatch.setattr("advface.attacks.mi_fgsm.run_mi_fgsm", fake_run)
    output = apply_attack(
        original, attack="mi_fgsm", app=object(),
        config=AttackConfig(name="mi_fgsm", eps=0.01, steps=3, alpha=0.002, momentum=0.9, seed=7),
    )
    assert isinstance(output, AttackOutput)
    assert output.name == "mi_fgsm"
    assert output.parameters["method"] == "mi_fgsm"
    assert output.parameters["momentum"] == pytest.approx(0.9)
    assert output.parameters["gradient_normalization"]
    assert output.parameters["random_start"] is False
    metrics = perturbation_metrics(original, output.adversarial_bgr)
    assert validate_linf(metrics["linf"], 2 / PIXEL_MAX)


def test_mi_fgsm_metadata_keeps_tensor_constraint_primary(monkeypatch):
    original = np.full((2, 2, 3), 100, dtype=np.uint8)
    adversarial = np.full_like(original, 103)

    def fake_run(*args, **kwargs):
        assert kwargs["return_metadata"] is True
        return adversarial, {
            "linf_tensor": 0.01,
            "linf_serialized": 3 / PIXEL_MAX,
            "linf_serialization_note": "serialized uint8 value may differ due to 8-bit quantization",
        }

    monkeypatch.setattr("advface.attacks.mi_fgsm.run_mi_fgsm", fake_run)
    output = apply_attack(
        original, attack="mi_fgsm", app=object(),
        config=AttackConfig(name="mi_fgsm", eps=0.01, steps=5),
    )
    assert validate_linf(output.parameters["linf_tensor"], 0.01)
    assert output.parameters["linf_serialized"] == pytest.approx(3 / PIXEL_MAX)
    assert output.parameters["linf_tensor"] <= 0.01


def test_registry_passes_constraint_and_tv_to_full_image_attacks(monkeypatch):
    original = np.full((2, 2, 3), 100, dtype=np.uint8)
    seen = {}
    def fake_pgd(*args, **kwargs):
        seen.update(kwargs)
        from advface.attacks.pgd import PgdResult
        return [PgdResult(eps=0.01, eps_255=2.55, steps=3, cosine=float('nan'), attacked_bgr=original, linf_tensor=0.01)]
    monkeypatch.setattr("advface.attacks.pgd.run_pgd_full", fake_pgd)
    output = apply_attack(original, attack="pgd_full", app=object(), config=AttackConfig(name="pgd_full", eps=0.01, steps=3, extra={"constraint": "landmark_superpixel_mask", "tv_weight": 0.0}))
    assert output.parameters["constraint"] == "landmark_superpixel_mask"
    assert seen["constraint"].name == "landmark_superpixel_mask"
