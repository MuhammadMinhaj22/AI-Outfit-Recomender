import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_API_KEY_ENV = "OPENWEATHER_API_KEY"

load_dotenv()


def fetch_weather(city: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    key = (api_key or os.getenv(OPENWEATHER_API_KEY_ENV, "")).strip()
    if not key:
        raise ValueError(
            f"{OPENWEATHER_API_KEY_ENV} is not set. Add it to environment or .env."
        )

    response = requests.get(
        OPENWEATHER_URL,
        params={"q": city, "appid": key, "units": "metric"},
        timeout=15,
    )

    if response.status_code == 404:
        raise ValueError(f"City '{city}' not found.")
    response.raise_for_status()

    payload = response.json()
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    weather = (payload.get("weather") or [{}])[0]

    timestamp_utc = int(payload.get("dt", datetime.now(timezone.utc).timestamp()))
    timezone_offset = int(payload.get("timezone", 0))
    local_dt = datetime.fromtimestamp(timestamp_utc + timezone_offset, tz=timezone.utc)

    return {
        "city": payload.get("name", city),
        "country": payload.get("sys", {}).get("country", ""),
        "temperature": float(main.get("temp", 0.0)),
        "humidity": float(main.get("humidity", 0.0)),
        "wind_speed": float(wind.get("speed", 0.0)),
        "weather_condition": str(weather.get("main", "Unknown")),
        "description": str(weather.get("description", "Unknown")),
        "local_month": local_dt.month,
        "observed_at": local_dt.isoformat(),
    }
