from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from apk2img_ml.cnn.tune_summary import (
    parse_sort_columns,
    render_optuna_summary_report,
    summarize_optuna_results,
)


class TuneSummaryTests(unittest.TestCase):
    def test_collects_metrics_from_local_best_eval_when_json_paths_are_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            tune_dir = root / "rgb_hi" / "cnn_optuna_no" / "20260510-005203"
            best_eval_dir = tune_dir / "best_eval" / "20260510-041145"
            best_eval_dir.mkdir(parents=True)

            self._write_json(
                tune_dir / "optuna_tuning_log.json",
                {
                    "timestamp": "20260510-005203",
                    "args": {
                        "data_root": "apk2img-ml-datasets/data/sdhash/image_rgb_hi",
                        "pretrained": False,
                        "model_candidates": ["resnet18"],
                        "batch_candidates": [16, 32, 64],
                        "optimizer_candidates": ["adam", "adamw"],
                        "lr_low": 1e-5,
                        "lr_high": 1e-3,
                        "weight_decay_candidates": [0.0, 1e-6, 1e-5, 1e-4],
                    },
                    "objective": "mean best validation accuracy across runs",
                    "best_trial": {
                        "number": 0,
                        "value": 0.956,
                        "params": {
                            "model": "resnet18",
                            "batch": 16,
                            "lr": 0.00021194800905984738,
                            "optimizer": "adam",
                            "weight_decay": 0.0,
                        },
                    },
                    "best_eval": {
                        "run_dir": "/home/IM25D009/git_workspace/results/rgb_hi/cnn_optuna_no/20260510-005203/best_eval/20260510-041145",
                        "log_json_path": "/home/IM25D009/git_workspace/results/rgb_hi/cnn_optuna_no/20260510-005203/best_eval/20260510-041145/train_log.json",
                        "summary": {
                            "test_acc_mean": 0.9615,
                            "macro_f1_mean": 0.9614988353397691,
                            "micro_f1_mean": 0.9615,
                            "weighted_f1_mean": 0.961498835339769,
                            "binary_malware_f1_mean": 0.9612870789341378,
                            "binary_benign_f1_mean": 0.9617105917454003,
                        },
                    },
                },
            )
            self._write_json(
                best_eval_dir / "train_log.json",
                {
                    "timestamp": "20260510-041145",
                    "args": {
                        "model": "resnet18",
                    },
                    "summary": {
                        "test_acc_mean": 0.9615,
                        "macro_f1_mean": 0.9614988353397691,
                        "micro_f1_mean": 0.9615,
                        "weighted_f1_mean": 0.961498835339769,
                        "binary_malware_f1_mean": 0.9612870789341378,
                        "binary_benign_f1_mean": 0.9617105917454003,
                    },
                },
            )

            report = summarize_optuna_results(root)
            self.assertEqual(len(report.records), 1)

            record = report.records[0]
            self.assertEqual(record.dataset, "rgb_hi")
            self.assertFalse(record.pretrained)
            self.assertEqual(record.model, "resnet18")
            self.assertEqual(record.tune_timestamp, "20260510-005203")
            self.assertEqual(record.best_eval_timestamp, "20260510-041145")
            self.assertEqual(record.best_eval_log_path, best_eval_dir / "train_log.json")
            self.assertAlmostEqual(record.best_objective_value or 0.0, 0.956)
            self.assertAlmostEqual(record.test_acc or 0.0, 0.9615)
            self.assertAlmostEqual(record.macro_f1 or 0.0, 0.9614988353397691)
            self.assertIn("batch", record.tuned_parameters)
            self.assertIn("lr", record.tuned_parameters)

            rendered = render_optuna_summary_report(report)
            self.assertIn("rgb_hi", rendered)
            self.assertIn("20260510-005203", rendered)
            self.assertIn("20260510-041145", rendered)
            self.assertIn("batch=16", rendered)
            self.assertIn("test_acc=0.961500", rendered)

    def test_latest_only_keeps_latest_timestamp_per_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            self._create_minimal_tune_run(root, "rgb_hi", "cnn_optuna_no", "20260510-005203")
            self._create_minimal_tune_run(root, "rgb_hi", "cnn_optuna_no", "20260510-105203")

            report = summarize_optuna_results(root, latest_only=True)
            self.assertEqual(len(report.records), 1)
            self.assertEqual(report.records[0].tune_timestamp, "20260510-105203")

    def test_table_only_renders_table_without_detail_blocks_and_sorts_by_selected_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            self._create_minimal_tune_run(
                root,
                "rgb_hi",
                "cnn_optuna_no",
                "20260510-005203",
                model="resnet18",
                test_acc=0.91,
                macro_f1=0.90,
                micro_f1=0.91,
            )
            self._create_minimal_tune_run(
                root,
                "rgb_hb",
                "cnn_optuna_pre",
                "20260510-105203",
                model="resnet50",
                test_acc=0.97,
                macro_f1=0.96,
                micro_f1=0.97,
            )

            report = summarize_optuna_results(root)
            rendered = render_optuna_summary_report(
                report,
                sort_by="test_acc",
                table_only=True,
            )

            lines = rendered.splitlines()
            self.assertTrue(lines[0].startswith("dataset"))
            self.assertNotIn("Found 2 Optuna runs", rendered)
            self.assertNotIn("[1]", rendered)
            self.assertNotIn("best_params", rendered)
            data_lines = [line for line in lines[2:] if line.strip()]
            self.assertTrue(data_lines[0].startswith("rgb_hb"))
            self.assertTrue(data_lines[1].startswith("rgb_hi"))

    def test_multi_key_sort_supports_priority_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results"
            self._create_minimal_tune_run(
                root,
                "rgb_hi",
                "cnn_optuna_pre",
                "20260510-005203",
                model="resnet50",
            )
            self._create_minimal_tune_run(
                root,
                "rgb_hi",
                "cnn_optuna_no",
                "20260510-005204",
                model="resnet18",
            )
            self._create_minimal_tune_run(
                root,
                "gray",
                "cnn_optuna_pre",
                "20260510-005205",
                model="resnet18",
            )

            report = summarize_optuna_results(root)
            sort_by = parse_sort_columns(["dataset,model", "pretrained"])
            rendered = render_optuna_summary_report(
                report,
                sort_by=sort_by,
                table_only=True,
            )

            data_lines = [line for line in rendered.splitlines()[2:] if line.strip()]
            self.assertTrue(data_lines[0].startswith("gray"))
            self.assertTrue(data_lines[1].startswith("rgb_hi  | false      | resnet18"))
            self.assertTrue(data_lines[2].startswith("rgb_hi  | true       | resnet50"))

    def test_parse_sort_columns_accepts_repeated_and_comma_separated_values(self) -> None:
        self.assertEqual(
            parse_sort_columns(["dataset,model", "pretrained"]),
            ("dataset", "model", "pretrained"),
        )
        self.assertEqual(parse_sort_columns(None), ("dataset",))

    def _create_minimal_tune_run(
        self,
        root: Path,
        dataset: str,
        pretrain_dir: str,
        timestamp: str,
        *,
        model: str = "resnet18",
        test_acc: float = 0.96,
        macro_f1: float = 0.95,
        micro_f1: float = 0.96,
        weighted_f1: float | None = None,
        malware_f1: float = 0.94,
        benign_f1: float = 0.96,
        val_acc: float = 0.95,
    ) -> None:
        tune_dir = root / dataset / pretrain_dir / timestamp
        best_eval_dir = tune_dir / "best_eval" / f"{timestamp}-eval"
        best_eval_dir.mkdir(parents=True)
        if weighted_f1 is None:
            weighted_f1 = macro_f1
        self._write_json(
            tune_dir / "optuna_tuning_log.json",
            {
                "timestamp": timestamp,
                "args": {
                    "pretrained": pretrain_dir.endswith("pre"),
                    "model_candidates": [model],
                    "batch_candidates": [16, 32],
                    "optimizer_candidates": ["adam", "adamw"],
                    "lr_low": 1e-5,
                    "lr_high": 1e-3,
                    "weight_decay_candidates": [0.0, 1e-4],
                },
                "best_trial": {
                    "value": val_acc,
                    "params": {
                        "model": model,
                        "batch": 16,
                        "lr": 1e-4,
                        "optimizer": "adam",
                        "weight_decay": 0.0,
                    },
                },
                "best_eval": {
                    "run_dir": str(best_eval_dir),
                    "summary": {
                        "test_acc_mean": test_acc,
                        "macro_f1_mean": macro_f1,
                        "micro_f1_mean": micro_f1,
                        "weighted_f1_mean": weighted_f1,
                        "binary_malware_f1_mean": malware_f1,
                        "binary_benign_f1_mean": benign_f1,
                    },
                },
            },
        )
        self._write_json(
            best_eval_dir / "train_log.json",
            {
                "timestamp": f"{timestamp}-eval",
                "args": {"model": model},
                "summary": {
                    "test_acc_mean": test_acc,
                    "macro_f1_mean": macro_f1,
                    "micro_f1_mean": micro_f1,
                    "weighted_f1_mean": weighted_f1,
                    "binary_malware_f1_mean": malware_f1,
                    "binary_benign_f1_mean": benign_f1,
                },
            },
        )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
