from download_models import download_files
download_files()

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feedback_store import init_feedback_db, save_feedback_record
from model import FEATURE_COLUMNS, OutfitRecommender, weather_to_ml_features
from weather import fetch_weather


app = FastAPI(title="Weather-Based Men's Outfit Recommender")
recommender = OutfitRecommender(model_path="outfit_model.joblib")
FEEDBACK_DB_PATH = "feedback.db"
VALID_FEEDBACK_LABELS = {"good", "bad"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    dataset_path: str = Field(default="dataset.csv")


class PredictRequest(BaseModel):
    city: str = Field(..., min_length=1)


class OutfitPayload(BaseModel):
    upper: str | None = None
    lower: str | None = None
    footwear: str | None = None
    extra: str | None = None
    upper_wear: str | None = None
    lower_wear: str | None = None
    accessories: str | None = None


class FeedbackRequest(BaseModel):
    city: str = Field(..., min_length=1)
    feedback: str = Field(..., min_length=1)
    weather: Dict[str, Any] | None = None
    features: Dict[str, Any] | None = None
    outfit: OutfitPayload | None = None
    weather_condition: str | None = None
    upper: str | None = None
    lower: str | None = None
    footwear: str | None = None
    extra: str | None = None


def _normalize_feedback_label(label: str) -> str:
    value = label.strip().lower()
    synonyms = {
        "positive": "good",
        "suitable": "good",
        "yes": "good",
        "negative": "bad",
        "not suitable": "bad",
        "no": "bad",
    }
    normalized = synonyms.get(value, value)
    if normalized not in VALID_FEEDBACK_LABELS:
        raise ValueError(
            "feedback must be one of: good, bad "
            "(or a compatible synonym like positive/negative)."
        )
    return normalized


def _normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in FEATURE_COLUMNS if key not in features]
    if missing:
        raise ValueError(
            f"features is missing required keys: {', '.join(missing)}"
        )

    return {
        "temperature": float(features["temperature"]),
        "humidity": float(features["humidity"]),
        "wind_speed": float(features["wind_speed"]),
        "weather_code": int(features["weather_code"]),
        "season": str(features["season"]),
        "temperature_bucket": str(features["temperature_bucket"]),
        "weather_type": str(features["weather_type"]),
    }


def _normalize_outfit(request: FeedbackRequest) -> Dict[str, str]:
    payload = (
        request.outfit.model_dump(exclude_none=True)
        if request.outfit is not None
        else {}
    )
    upper = payload.get("upper") or payload.get("upper_wear") or request.upper
    lower = payload.get("lower") or payload.get("lower_wear") or request.lower
    footwear = payload.get("footwear") or request.footwear
    extra = payload.get("extra") or payload.get("accessories") or request.extra

    outfit = {
        "upper": str(upper).strip() if upper is not None else "",
        "lower": str(lower).strip() if lower is not None else "",
        "footwear": str(footwear).strip() if footwear is not None else "",
        "extra": str(extra).strip() if extra is not None else "",
    }
    missing = [key for key, value in outfit.items() if not value]
    if missing:
        raise ValueError(
            f"outfit is missing required fields: {', '.join(missing)}"
        )
    return outfit


def _resolve_weather_and_features(
    request: FeedbackRequest,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    weather_data = dict(request.weather) if request.weather else {}
    if request.weather_condition and "weather_condition" not in weather_data:
        weather_data["weather_condition"] = request.weather_condition

    if request.features:
        features = _normalize_features(dict(request.features))
    elif weather_data:
        features = weather_to_ml_features(weather_data)
    else:
        weather_data = fetch_weather(request.city)
        features = weather_to_ml_features(weather_data)

    if not weather_data:
        weather_data = {
            "city": request.city,
            "temperature": features["temperature"],
            "humidity": features["humidity"],
            "wind_speed": features["wind_speed"],
            "weather_condition": request.weather_condition or features["weather_type"],
            "description": "Unknown",
            "country": "",
        }

    # Keep numeric weather values aligned with stored features for retraining prep.
    weather_data.setdefault("temperature", features["temperature"])
    weather_data.setdefault("humidity", features["humidity"])
    weather_data.setdefault("wind_speed", features["wind_speed"])
    weather_data.setdefault("weather_condition", request.weather_condition or "Unknown")
    weather_data.setdefault("description", "Unknown")

    return weather_data, features


@app.on_event("startup")
def startup_bootstrap() -> None:
    init_feedback_db(FEEDBACK_DB_PATH)

    try:
        recommender.load()
    except FileNotFoundError:
        default_dataset = Path("dataset.csv")
        if default_dataset.exists():
            try:
                recommender.train(str(default_dataset))
            except Exception:
                # Keep API startup alive; user can retrain via POST /train.
                pass


@app.get("/weather")
def get_weather(city: str) -> dict:
    try:
        return fetch_weather(city)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}") from exc


@app.post("/train")
def train_model(request: TrainRequest) -> dict:
    try:
        info = recommender.train(request.dataset_path)
        return {"status": "trained", **info}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@app.post("/predict")
def predict_outfit(request: PredictRequest) -> dict:
    try:
        weather_data = fetch_weather(request.city)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}") from exc

    try:
        recommendation, features = recommender.predict(weather_data)
    except FileNotFoundError:
        default_dataset = Path("dataset.csv")
        if default_dataset.exists():
            recommender.train(str(default_dataset))
            recommendation, features = recommender.predict(weather_data)
        else:
            raise HTTPException(
                status_code=503,
                detail="Model not trained. Add dataset.csv and call POST /train.",
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "upper": recommendation["upper"],
        "lower": recommendation["lower"],
        "footwear": recommendation["footwear"],
        "extra": recommendation["extra"],
        "city": weather_data["city"],
        "weather": weather_data,
        "features": features,
    }


@app.post("/feedback")
def submit_feedback(request: FeedbackRequest) -> dict:
    try:
        feedback_label = _normalize_feedback_label(request.feedback)
        outfit = _normalize_outfit(request)
        weather_data, features = _resolve_weather_and_features(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not prepare feedback context: {exc}",
        ) from exc

    try:
        record_id = save_feedback_record(
            db_path=FEEDBACK_DB_PATH,
            city=str(weather_data.get("city") or request.city),
            weather=weather_data,
            features=features,
            outfit=outfit,
            feedback_label=feedback_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {exc}") from exc

    return {
        "status": "saved",
        "record_id": record_id,
        "feedback_label": feedback_label,
        "retraining_ready": True,
    }
