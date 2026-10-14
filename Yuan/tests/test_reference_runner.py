"""Dependency-free checks for pairing, denominators and input handling."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_reference.py"
spec = importlib.util.spec_from_file_location("reference_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReferenceRunnerTests(unittest.TestCase):
    def test_asr_excludes_clean_invalid_pairs_and_separates_models(self):
        rows = [
            {"mode": "pair", "source_clean_valid": True, "source_boundary_crossed": True,
             "source_drop": 0.5, "victim_clean_valid": True, "victim_boundary_crossed": False, "victim_drop": 0.2},
            {"mode": "pair", "source_clean_valid": True, "source_boundary_crossed": False,
             "source_drop": 0.1, "victim_clean_valid": False, "victim_boundary_crossed": False, "victim_drop": 0.8},
        ]
        result = runner.aggregate(rows, "pair")
        self.assertEqual(result["source"]["asr"], 0.5)
        self.assertEqual(result["victim"]["eligible"], 1)
        self.assertEqual(result["victim"]["asr"], 0.0)
        self.assertEqual(result["victim"]["mean_cosine_drop"], 0.2)

    def test_empty_results_have_no_success_rate(self):
        self.assertIsNone(runner.aggregate([], "self")["source"]["asr"])

    def test_duplicate_identities_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack_dev.csv"
            path.write_text("identity_id,reference_image,probe_image\nx,a,b\nx,c,d\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                runner.read_pairs(path)

    def test_read_bom_split_and_resolve_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack_dev.csv"
            path.write_text("identity_id,reference_image,probe_image\nx,images/a.jpg,images/b.jpg\n", encoding="utf-8-sig")
            rows = runner.read_pairs(path)
            self.assertEqual(rows[0]["identity_id"], "x")
            self.assertEqual(runner.input_path(Path(directory), rows[0]["reference_image"]),
                             Path(directory) / "images/a.jpg")


if __name__ == "__main__":
    unittest.main()
