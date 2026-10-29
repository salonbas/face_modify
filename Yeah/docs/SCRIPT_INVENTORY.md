# Script and architecture audit

Audited 2026-10-29. Paths are relative to the repository root. `F` = formal or frozen protocol, `D` = diagnostic / smoke, `H` = historical or general exploratory, `S` = setup / specification. A formal label does not imply the five-model 100-person benchmark ran. Entry point is `python scripts/<name>.py` unless noted. Inputs and outputs below are the principal artifacts; use each CLI's `--help` for optional flags. No scripts were moved or deleted during this audit.

| Script | Role; main input → output | Upstream caller / downstream dependency | Current formal use |
| --- | --- | --- | --- |
| `setup_research_assets.py` | S; upstream LFW and model archives, manifest → ignored local assets | Manual setup → environment check, all model CLIs | Required setup; author weights manual |
| `install_research_dependencies.sh` | S; pinned requirements → isolated venv packages | Manual setup → all Python CLIs | Required setup |
| `setup_five_model_sources.sh` | S; pinned author Git commits → ignored AdaFace/MagFace source modules | Manual setup → author victim loaders | Required for v2 smoke |
| `check_research_environment.py` | S; assets, splits, calibration, v3 summary → completeness report | Manual or smoke gate → researchers | Yes |
| `generate_research_specs.py` | S; frozen JSON → two docs | Maintainer → `CALIBRATION_AND_THRESHOLDS`, `EXPERIMENT_CONFIG` | Documentation guard |
| `prepare_lfw_dataset.py` | H; sklearn LFW cache → manifest | Dataset setup → split creators | Historical preparation; manifest frozen |
| `prepare_lfw_attack_splits.py` | F provenance; LFW manifest/pairs → v1 attack split CSVs | Manual freeze → v3 runners | Inputs frozen; do not rerun casually |
| `prepare_lfw_v2_protocol.py` | F provenance; old splits/calibration pairs → v2 100/228 split and audit | Manual freeze → v2 calibration/preflight | Inputs frozen; do not rerun casually |
| `calibrate_threshold.py` | F provenance; official LFW train/test pairs, ArcFace/FaceNet → v1 thresholds/scores | Manual calibration → v1 verification CLIs | V3 authority |
| `calibrate_facenet_threshold.py` | H compatibility wrapper; FaceNet calibration arguments → v1 artifacts | Manual → `calibrate_threshold.py` | Kept for existing commands |
| `calibrate_lfw_v2_victims.py` | F provenance; v2 calibration pairs, five-model weights → v2 thresholds/metadata | Manual → v2 eligibility/preflight | V2 authority |
| `check_lfw_five_model_clean_eligibility.py` | D; Development 100, v2 thresholds → clean eligibility JSON | Manual → v2 preflight/smoke interpretation | Clean-only evidence |
| `preflight_lfw_multi_victim_v1.py` | D; split, weights, thresholds, timing → preflight JSON | Manual → five-model gate | Earlier fail-closed gate |
| `run_lfw_five_model_benchmark.py` | F gate; same inputs → READY/NOT READY | Manual → proposed full executor | Currently fail-closed; no full output |
| `smoke_lfw_five_model_pgd_mask.py` | D; ≤2 Development pairs, five models → smoke PNG/JSON | Manual → v2 engineering validation | Smoke only |
| `run_lfw_dev_v2.py` | F; frozen 20 pairs, PGD/MI-FGSM, mask option → serialized PNG/results | Manual → v3 baseline/aggregation | Yes |
| `run_lfw_two_phase_v4.py` | F; frozen 20 pairs → generation records, then evaluations/results | Manual; called by own subprocess → v3 aggregation | Yes; resumable method runner |
| `aggregate_lfw_four_method_v3.py` | F; four method results → summary CSV/JSON/HTML | Manual after runs → mechanism/report | Yes; no attack generation |
| `analyze_lfw_four_method_mechanisms.py` | F analysis; v3 per-identity results → mechanism JSON/plots | Manual → meeting report | Formal analysis, read-only inputs |
| `run_momentum_mask_v2.py` | D; two allowlisted pairs, μ=0/0.5 → separate PNG/evaluation | Manual → momentum comparison | Exploratory, separate from v3 |
| `evaluate_lfw_perceptual_v2.py` | F provenance; five fixed pairs and serialized PNGs → perceptual table/report | Manual → v1 comparison | Historical frozen five-person protocol |
| `evaluate_perceptual_v1.py` | H; selected serialized image pairs → perceptual records | Manual → exploratory analysis | General evaluator; do not pool |
| `run_verification_robustness.py` | F provenance; five v1 pairs, ArcFace PGD → robustness CSV | Manual → perceptual v1/v2 | Historical source authority |
| `run_transfer_verification.py` | F provenance; five v1 pairs, PGD, FaceNet → transfer CSV/PNG | Manual → perceptual v2 | Historical regenerated PGD artifact |
| `run_mi_fgsm_transfer_verification.py` | F provenance; five v1 pairs, MI-FGSM → transfer CSV/PNG | Manual → perceptual v2 | Historical method authority |
| `run_lfw_mask_smoke.py` | D; five v1 pairs → mask smoke results | Manual → mask validation | Smoke only |
| `run_lfw_gradient_trajectory_diagnostics.py` | D; frozen 20 pairs → trajectory/checkpoints | Manual → mechanism diagnosis | Diagnostic, not formal rate |
| `benchmark_diagnostic_performance.py` | D; one pair/model → timing/checkpoints | Manual → performance gate | Diagnostic only |
| `run_benchmark.py` | H; generic benchmark manifest/grid → benchmark results | Manual → `advface.benchmark.runner` | Not v3 LFW authority |
| `prepare_benchmark_dataset.py` | H; raw images → generic benchmark manifest/copies | Manual → generic benchmark | Not v3 LFW authority |
| `run_attack.py` | H; one image/attack config → attack images/metrics | Manual → `advface.attacks` | Demonstration only |
| `run_transfer.py` | H; one image, surrogate/victim → transfer result | Manual → `advface.experiments` | Exploratory only |
| `run_gaussian.py` | H; one image/epsilon sweep → CSV/plot | Manual → Gaussian attack | Exploratory only |
| `verify_attack.py` | H; original/attacked image → similarity result | Manual → model embedder | Exploratory only |
| `generate_meeting_report.py` | H report; existing artifacts → dated HTML report | Manual → presentation | Report provenance |

## Package and artifact layout

`advface/attacks` owns attack algorithms, registry, projection and pixel clipping. `advface/models` owns ArcFace, FaceNet and three new victim adapters; `advface/constraints` owns the landmark/SLIC mask. `advface/calibration.py` owns empirical threshold selection and FAR/TAR calculations. `advface/evaluation` owns similarity, verification, perturbation and five-model eligible-set aggregation. `advface/experiments` and `advface/benchmark` support generic runs and checkpoints. `advface/paths.py` and `advface/image_io.py` own paths and image loading. `scripts/` are CLI orchestration; `tests/` cover attack, calibration, benchmark, model alignment, paths and five-model summaries.

`data/datasets/lfw/` contains tracked manifest/pairs/splits and ignored raw images. `models/checkpoint_manifest.json` is tracked model provenance; binary weights and author source copies are ignored. `results/calibration/` and curated `results/transfer/lfw_attack_dev_20_four_method_v3/` contain formal tabular/JSON evidence. `results/smoke`, diagnostic checkpoints, raw adversarial PNGs, embedding arrays and caches are ignored. `docs/` contains protocol/specification and reports. `archive/` already holds legacy docs; no further archive move was justified without caller-level regression.

## Duplication and boundaries

ArcFace/FaceNet initialization, face preprocessing and embedding are already centralized in `advface/models`; runners call `load_embedder` and `create_face_app`. Threshold loading for v1 is centralized in `advface/evaluation/verification.py`; v2 five-model eligibility and smoke now share `load_v2_thresholds` in `advface/evaluation/multi_victim.py` because the older loader requires v1 EER and exploratory points. Threshold **selection** is shared through `advface/calibration.py`. Mask construction, attack methods and projection are package functions; separate runners mainly encode distinct study cohorts and output contracts. V1/v2 split builders are distinct because v2 explicitly removes calibration overlap. Generic benchmark aggregation, v3 aggregation and five-model eligible aggregation use different denominators; they should not be merged by filename similarity. Checkpoint handling is centralized for generic benchmarks, while formal LFW runners implement append-only per-identity records. A future refactor can share record I/O after fixed-output regression fixtures exist; no further duplicate was removed in this pass.

The v3 `run_lfw_dev_v2.py` and `run_lfw_two_phase_v4.py` overlap in attack generation but serve different resumability/evaluation contracts and existing artifact provenance. `run_transfer_verification.py` and `run_mi_fgsm_transfer_verification.py` likewise encode different frozen methods. Retain them as compatibility/provenance entries. New work should use package helpers and choose a single formal CLI per protocol rather than adding another runner.
