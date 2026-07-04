import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = ["temperature", "humidity", "weather_type", "season"]
TARGET_COLUMNS = ["upper_wear", "lower_wear", "footwear", "accessories"]


def train_model(dataset_path: str, model_path: str) -> None:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    df = pd.read_csv(dataset_file)
    if df.empty:
        raise ValueError("Dataset is empty.")

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
            + ". Expected columns: "
            + ", ".join(required_columns)
        )

    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()

    x["temperature"] = pd.to_numeric(x["temperature"], errors="coerce")
    x["humidity"] = pd.to_numeric(x["humidity"], errors="coerce")

    if x["temperature"].isna().all() or x["humidity"].isna().all():
        raise ValueError("temperature/humidity columns must contain numeric values.")

    x["temperature"] = x["temperature"].fillna(x["temperature"].median())
    x["humidity"] = x["humidity"].fillna(x["humidity"].median())
    x["weather_type"] = x["weather_type"].fillna("unknown").astype(str).str.strip()
    x["season"] = x["season"].fillna("unknown").astype(str).str.strip()
    x.loc[x["weather_type"].str.lower().isin({"", "nan", "none", "<na>"}), "weather_type"] = "unknown"
    x.loc[x["season"].str.lower().isin({"", "nan", "none", "<na>"}), "season"] = "unknown"

    for col in TARGET_COLUMNS:
        y[col] = y[col].fillna("unknown").astype(str).str.strip()
        y.loc[y[col].str.lower().isin({"", "nan", "none", "<na>"}), col] = "unknown"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["temperature", "humidity"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["weather_type", "season"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=1,
                    )
                ),
            ),
        ]
    )

    model.fit(x, y)

    output = {"model": model, "features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS}
    with open(model_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Model trained with {len(df)} rows.")
    print(f"Saved model to: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a RandomForest outfit recommender model."
    )
    parser.add_argument(
        "--dataset",
        default="dataset.csv",
        help="Path to training dataset CSV (default: dataset.csv).",
    )
    parser.add_argument(
        "--output",
        default="outfit_model.pkl",
        help="Output model path (default: outfit_model.pkl).",
    )
    args = parser.parse_args()

    train_model(args.dataset, args.output)


if __name__ == "__main__":
    main()
