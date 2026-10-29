from advface.evaluation.multi_victim import MODELS, summarize


def test_clean_failure_never_counts_as_attack_success():
    thresholds = {name: 0.5 for name in MODELS}
    rows = [
        {"identity_id": "A", "models": {name: {"clean_cosine": 0.8, "eligible": True,
                                          "adv_cosine": 0.3, "success": True} for name in MODELS}},
        {"identity_id": "B", "models": {name: {"clean_cosine": 0.3, "eligible": False,
                                          "adv_cosine": 0.2, "success": False} if name == "facenet" else
                                     {"clean_cosine": 0.8, "eligible": True,
                                      "adv_cosine": 0.7, "success": False} for name in MODELS}},
    ]
    report = summarize(rows, thresholds)
    assert report["original_denominator"] == 2
    assert report["joint_eligible"] == 1
    assert report["models"]["facenet"]["clean_tar"] == 0.5
    assert report["models"]["facenet"]["conditional_denominator"] == 1
    assert report["models"]["facenet"]["paired_asr"] == 1.0


def test_detection_failure_keeps_original_denominator():
    thresholds = {name: 0.5 for name in MODELS}
    row = {"identity_id": "A", "models": {name: {"clean_cosine": 0.8, "eligible": True}
                                         for name in MODELS}}
    row["models"]["adaface"] = {"clean_cosine": None, "eligible": False, "error": "no face"}
    result = summarize([row], thresholds)
    assert result["original_denominator"] == 1
    assert result["joint_eligible"] == 0
    assert result["models"]["adaface"]["clean_errors"] == 1
    assert result["models"]["adaface"]["clean_tar"] == 0
