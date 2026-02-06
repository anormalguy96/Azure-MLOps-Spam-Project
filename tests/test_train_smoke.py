from __future__ import annotations

from pathlib import Path

from src.training.train import TrainConfig, main


def test_train_smoke(tmp_path: Path) -> None:
    model_out = tmp_path / "model"
    cfg = TrainConfig(
        data_path=Path("data/spam_sample.csv"),
        output_dir=model_out,
        test_size=0.2,
        random_state=42,
        max_features=2000,
        ngram_max=2,
        c=1.0,
        max_iter=100,
    )
    rc = main(cfg)
    assert rc == 0
    assert (model_out / "MLmodel").exists()
    assert (model_out / "metrics.json").exists()
