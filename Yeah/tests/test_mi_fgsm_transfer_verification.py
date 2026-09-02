from pathlib import Path


def test_mi_fgsm_transfer_runner_is_development_only_and_preserves_protocol():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/run_mi_fgsm_transfer_verification.py").read_text(encoding="utf-8")
    assert 'choices=["development"]' in script
    assert 'ATTACKS["mi_fgsm"].apply' in script
    assert 'AttackConfig(name="mi_fgsm", eps=EPS, steps=steps, momentum=MOMENTUM)' in script
    assert 'victim_gradient_participation": False' in script
    assert 'victim = load_embedder("facenet_vggface2")' in script
    assert 'source_crossed and victim_valid and victim_adv <' in script
    assert 'linf_tensor) > EPS + TOLERANCE' in script
    assert 'comparison_pgd_vs_mi_fgsm.csv' in script
