from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMNS = ["upper", "lower", "footwear", "extra"]
FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "weather_code",
    "season",
    "temperature_bucket",
    "weather_type",
]

COLUMN_ALIASES = {
    "temperature": ["temperature", "temp", "temp_c", "temperature_c"],
    "humidity": ["humidity", "humid"],
    "wind_speed": ["wind_speed", "wind", "windspeed", "wind_kph", "wind_mps"],
    "weather_condition": [
        "weather_condition",
        "weather",
        "condition",
        "weather_main",
        "weather_type",
    ],
    "season": ["season", "season_name"],
    "upper": ["upper", "upper_wear", "top", "shirt"],
    "lower": ["lower", "lower_wear", "bottom", "pants"],
    "footwear": ["footwear", "shoes", "shoe"],
    "extra": ["extra", "accessories", "accessory"],
}


def _temperature_bucket(temp_c: float) -> str:
    if temp_c < 10:
        return "very_cold"
    if temp_c < 20:
        return "cool"
    if temp_c <= 30:
        return "warm"
    return "hot"


def _encode_weather_condition(condition: str) -> Tuple[str, int]:
    label = (condition or "").strip().lower()

    if any(word in label for word in ["rain", "drizzle", "shower"]):
        return "rainy", 0
    if any(word in label for word in ["cloud"]):
        return "cloudy", 1
    if any(word in label for word in ["clear", "sun"]):
        return "sunny", 2
    if any(word in label for word in ["snow", "sleet", "ice"]):
        return "snowy", 3
    if any(word in label for word in ["storm", "thunder", "squall"]):
        return "stormy", 4
    if any(word in label for word in ["mist", "fog", "haze", "smoke", "dust"]):
        return "misty", 5
    if any(word in label for word in ["wind", "breez", "gust"]):
        return "windy", 6
    return "other", 7


def _normalize_season(season: str) -> str:
    value = (season or "").strip().lower()
    if value in {"autumn", "fall"}:
        return "fall"
    if value in {"spring", "summer", "winter"}:
        return value
    return "unknown"


def _month_to_season(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "fall"


def _default_wind_speed_from_condition(condition: str) -> float:
    weather_type, _ = _encode_weather_condition(condition)
    defaults = {
        "sunny": 2.8,
        "cloudy": 3.2,
        "rainy": 4.6,
        "snowy": 3.9,
        "stormy": 7.5,
        "misty": 2.4,
        "windy": 6.8,
        "other": 3.0,
    }
    return defaults.get(weather_type, 3.0)


def weather_to_ml_features(weather: Dict[str, Any]) -> Dict[str, Any]:
    temperature = float(weather.get("temperature", 0.0))
    humidity = float(weather.get("humidity", 0.0))
    wind_speed = float(weather.get("wind_speed", 0.0))

    weather_condition = str(weather.get("weather_condition", "other"))
    weather_type, weather_code = _encode_weather_condition(weather_condition)

    season = weather.get("season")
    if season:
        season_value = _normalize_season(str(season))
    else:
        local_month = weather.get("local_month")
        if local_month is None:
            local_month = datetime.utcnow().month
        season_value = _month_to_season(int(local_month))

    return {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "weather_code": weather_code,
        "season": season_value,
        "temperature_bucket": _temperature_bucket(temperature),
        "weather_type": weather_type,
    }


def _standardize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = {col.strip().lower(): col for col in df.columns}
    rename_map: Dict[str, str] = {}
    missing: List[str] = []
    required_columns = {
        "temperature",
        "humidity",
        "season",
        "upper",
        "lower",
        "footwear",
        "extra",
    }

    for canonical, aliases in COLUMN_ALIASES.items():
        found = next((a for a in aliases if a in lower_to_original), None)
        if not found:
            if canonical in required_columns:
                missing.append(canonical)
            continue
        rename_map[lower_to_original[found]] = canonical

    if missing:
        raise ValueError(
            f"dataset.csv is missing required columns: {', '.join(missing)}. "
            f"Available columns: {', '.join(df.columns)}"
        )

    standardized = df.rename(columns=rename_map).copy()

    if "weather_condition" not in standardized.columns:
        standardized["weather_condition"] = "unknown"
    standardized["weather_condition"] = standardized["weather_condition"].astype(str).str.strip()
    standardized.loc[standardized["weather_condition"] == "", "weather_condition"] = "unknown"

    if "wind_speed" not in standardized.columns:
        standardized["wind_speed"] = standardized["weather_condition"].map(
            _default_wind_speed_from_condition
        )

    standardized = standardized[list(COLUMN_ALIASES.keys())].copy()

    for col in ["temperature", "humidity", "wind_speed"]:
        standardized[col] = pd.to_numeric(standardized[col], errors="coerce")
        if standardized[col].isna().all():
            raise ValueError(f"Column '{col}' has no valid numeric values.")
        standardized[col] = standardized[col].fillna(standardized[col].median())

    for col in ["weather_condition", "season"] + TARGET_COLUMNS:
        standardized[col] = standardized[col].fillna("unknown").astype(str).str.strip()
        standardized.loc[
            standardized[col].str.lower().isin({"", "nan", "none", "<na>"}),
            col,
        ] = "unknown"

    standardized["season"] = standardized["season"].map(_normalize_season)
    return standardized


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    weather_encoded = df["weather_condition"].map(_encode_weather_condition)
    weather_type = weather_encoded.map(lambda item: item[0])
    weather_code = weather_encoded.map(lambda item: item[1])

    features = pd.DataFrame(
        {
            "temperature": df["temperature"].astype(float),
            "humidity": df["humidity"].astype(float),
            "wind_speed": df["wind_speed"].astype(float),
            "weather_code": weather_code.astype(int),
            "season": df["season"].astype(str),
            "temperature_bucket": df["temperature"].map(_temperature_bucket),
            "weather_type": weather_type.astype(str),
        }
    )
    return features[FEATURE_COLUMNS]


def _build_pipeline() -> Pipeline:
    numeric_features = ["temperature", "humidity", "wind_speed", "weather_code"]
    categorical_features = ["season", "temperature_bucket", "weather_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    classifier = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=1,
        )
    )

    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


class OutfitRecommender:
    def __init__(self, model_path: str = "outfit_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.pipeline: Pipeline | None = None
        self.target_columns: List[str] = TARGET_COLUMNS.copy()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        saved = joblib.load(self.model_path)
        self.pipeline = saved["pipeline"]
        self.target_columns = saved.get("target_columns", TARGET_COLUMNS.copy())

    def train(self, dataset_path: str = "dataset.csv") -> Dict[str, Any]:
        dataset = Path(dataset_path)
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset}")

        raw_df = pd.read_csv(dataset)
        if raw_df.empty:
            raise ValueError("dataset.csv is empty.")

        data = _standardize_dataset_columns(raw_df)
        x_train = _build_feature_frame(data)
        y_train = data[TARGET_COLUMNS].astype(str)

        pipeline = _build_pipeline()
        pipeline.fit(x_train, y_train)

        self.pipeline = pipeline
        model_blob = {
            "pipeline": pipeline,
            "target_columns": TARGET_COLUMNS,
            "trained_rows": len(data),
        }
        joblib.dump(model_blob, self.model_path)

        return {
            "model_path": str(self.model_path),
            "dataset_path": str(dataset),
            "trained_rows": len(data),
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
        }

    def predict(self, weather: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        if self.pipeline is None:
            self.load()

        features = weather_to_ml_features(weather)
        x_input = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        prediction = self.pipeline.predict(x_input)[0]

        recommendation = {
            col: str(value) for col, value in zip(self.target_columns, prediction)
        }
        return recommendation, features
