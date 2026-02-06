from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LOG = logging.getLogger("training")


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    output_dir: Path
    test_size: float
    random_state: int
    max_features: int
    ngram_max: int
    c: float
    max_iter: int


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_dataset(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError('CSV must contain columns: "text", "label"')
    x = df["text"].astype(str)
    y = df["label"].astype(int)
    return x, y


def build_model(cfg: TrainConfig) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=cfg.max_features,
                    ngram_range=(1, cfg.ngram_max),
                ),
            ),
            ("clf", LogisticRegression(C=cfg.c, max_iter=cfg.max_iter, n_jobs=1)),
        ]
    )


def evaluate(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    return {"accuracy": acc, "f1": f1}


def save_mlflow_model(model: Pipeline, output_dir: Path) -> None:
    """Save an MLflow model folder suitable for Azure ML model registration.

    In Azure ML command jobs, the output folder path is available via:
      AZUREML_OUTPUT_MODEL_OUTPUT

    We register this folder as a model asset with type "mlflow_model" and deploy it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    input_example = pd.DataFrame({"text": ["free prize now"]})

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(output_dir),
        input_example=input_example,
    )


def main(cfg: TrainConfig) -> int:
    configure_logging()
    LOG.info("Loading data from %s", cfg.data_path)
    x, y = load_dataset(cfg.data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    model = build_model(cfg)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "max_features": cfg.max_features,
                "ngram_max": cfg.ngram_max,
                "C": cfg.c,
                "max_iter": cfg.max_iter,
                "model_type": "LogisticRegression",
            }
        )

        LOG.info("Training...")
        model.fit(x_train, y_train)

        metrics = evaluate(model, x_test, y_test)
        mlflow.log_metrics(metrics)
        LOG.info("Metrics: %s", metrics)

        save_mlflow_model(model, cfg.output_dir)
        (cfg.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    LOG.info("Done. Model saved to %s", cfg.output_dir)
    return 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a spam detector with MLflow logging.")
    parser.add_argument("--data", type=Path, required=True, help="Path to spam_sample.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("AZUREML_OUTPUT_MODEL_OUTPUT", "artifacts/model")),
        help="Where to save the MLflow model folder (default uses AZUREML_OUTPUT_MODEL_OUTPUT in Azure ML).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)

    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        c=args.c,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
