import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def init_feedback_db(db_path: str = "feedback.db") -> None:
    database = Path(db_path)
    with sqlite3.connect(database) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                city TEXT NOT NULL,
                country TEXT,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                weather_condition TEXT,
                weather_description TEXT,
                season TEXT,
                weather_type TEXT,
                weather_code INTEGER,
                temperature_bucket TEXT,
                upper TEXT NOT NULL,
                lower TEXT NOT NULL,
                footwear TEXT NOT NULL,
                extra TEXT NOT NULL,
                feedback_label TEXT NOT NULL,
                weather_json TEXT NOT NULL,
                features_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at
            ON feedback_records(created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_label
            ON feedback_records(feedback_label)
            """
        )
        conn.commit()


def save_feedback_record(
    *,
    db_path: str,
    city: str,
    weather: Dict[str, Any],
    features: Dict[str, Any],
    outfit: Dict[str, str],
    feedback_label: str,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    database = Path(db_path)

    with sqlite3.connect(database) as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback_records (
                created_at,
                city,
                country,
                temperature,
                humidity,
                wind_speed,
                weather_condition,
                weather_description,
                season,
                weather_type,
                weather_code,
                temperature_bucket,
                upper,
                lower,
                footwear,
                extra,
                feedback_label,
                weather_json,
                features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                city,
                weather.get("country"),
                weather.get("temperature"),
                weather.get("humidity"),
                weather.get("wind_speed"),
                weather.get("weather_condition"),
                weather.get("description"),
                features.get("season"),
                features.get("weather_type"),
                features.get("weather_code"),
                features.get("temperature_bucket"),
                outfit["upper"],
                outfit["lower"],
                outfit["footwear"],
                outfit["extra"],
                feedback_label,
                json.dumps(weather),
                json.dumps(features),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

