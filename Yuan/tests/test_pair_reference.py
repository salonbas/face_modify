"""Exercise A's independent alignment with small fake models; no weights needed."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("insightface")

from insightface.utils import face_align
from advface.attacks import mi_fgsm


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("di_probability", [0.0, 1.0])
@pytest.mark.parametrize("ti_kernel_size", [1, 5])
def test_pair_reference_uses_a_landmarks_and_preserves_budget(monkeypatch, masked, di_probability, ti_kernel_size):
    b = np.full((112, 112, 3), 100, dtype=np.uint8)
    a = np.full_like(b, 180)
    b_points = np.full((5, 2), 10, dtype=np.float32)
    a_points = np.full((5, 2), 20, dtype=np.float32)
    calls = []

    class App:
        def get(self, image):
            return [SimpleNamespace(det_score=1, kps=a_points if image is a else b_points)]

    class Model(torch.nn.Module):
        def forward(self, image):
            mean = image.mean(dim=(1, 2, 3))
            return torch.stack((mean, torch.ones_like(mean)), dim=1)

    def crop_b(image, landmark, image_size):
        assert image is b
        np.testing.assert_array_equal(landmark, b_points)
        return image.copy(), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    def crop_a(image, landmark, image_size):
        assert image is a
        np.testing.assert_array_equal(landmark, a_points)
        calls.append("A independently aligned")
        return image.copy()

    monkeypatch.setattr(face_align, "norm_crop2", crop_b)
    monkeypatch.setattr(face_align, "norm_crop", crop_a)
    monkeypatch.setattr(mi_fgsm, "load_arcface_torch", lambda device: Model())
    mask = None
    if masked:
        mask = np.zeros(b.shape[:2], dtype=np.float32)
        mask[28:84, 28:84] = 1
    smoothed = []
    original_smoother = mi_fgsm.translation_invariant_gradient
    def checked_smoother(gradient, **kwargs):
        if masked:
            # TI receives the raw full-image gradient; mask is applied after TI.
            outside = torch.from_numpy(mask == 0)
            assert gradient[0, :, outside].abs().sum() > 0
        smoothed.append(True)
        return original_smoother(gradient, **kwargs)
    monkeypatch.setattr(mi_fgsm, "translation_invariant_gradient", checked_smoother)
    modified, metadata = mi_fgsm.run_mi_fgsm(b, 0.04, App(), steps=2,
        device="cpu", reference_image=a, return_metadata=True, precomputed_mask=mask,
        di_probability=di_probability, di_seed=0, ti_kernel_size=ti_kernel_size)
    assert len(smoothed) == (2 if ti_kernel_size > 1 else 0)
    assert calls == ["A independently aligned"]
    assert metadata["reference_mode"] == "pair"
    assert metadata["ti_kernel_size"] == ti_kernel_size
    assert metadata["initial_gradient_l1"] > 0
    assert metadata["final_objective_cosine"] < metadata["initial_objective_cosine"]
    assert metadata["linf_tensor"] <= 0.04 + 1e-6
    assert modified.shape == b.shape
    assert np.any(modified != b)
    if masked:
        np.testing.assert_array_equal(modified[mask == 0], b[mask == 0])
    np.testing.assert_array_equal(a, np.full_like(a, 180))


def test_self_reference_does_not_align_a(monkeypatch):
    b = np.full((112, 112, 3), 100, dtype=np.uint8)
    class Model(torch.nn.Module):
        def forward(self, image):
            mean = image.mean(dim=(1, 2, 3))
            return torch.stack((mean, torch.ones_like(mean)), dim=1)
    app = SimpleNamespace(get=lambda image: [SimpleNamespace(det_score=1, kps=np.zeros((5, 2)))])
    monkeypatch.setattr(face_align, "norm_crop2", lambda *args, **kwargs:
        (b.copy(), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)))
    def fail(*args, **kwargs):
        raise AssertionError("self-reference must not call A alignment")
    monkeypatch.setattr(face_align, "norm_crop", fail)
    monkeypatch.setattr(mi_fgsm, "load_arcface_torch", lambda device: Model())
    _, metadata = mi_fgsm.run_mi_fgsm(b, 0.04, app, steps=2, device="cpu", return_metadata=True)
    assert metadata["reference_mode"] == "self"
