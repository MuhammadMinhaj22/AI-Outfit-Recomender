const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("outfit-form");
const cityInput = document.getElementById("city-input");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const feedbackStatusEl = document.getElementById("feedback-status");
const feedbackGoodBtn = document.getElementById("feedback-good");
const feedbackBadBtn = document.getElementById("feedback-bad");

let lastPrediction = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "Loading..." : "Get Outfit";
}

function renderResult(payload) {
  const weather = payload.weather || {};

  document.getElementById("weather-city").textContent =
    weather.city && weather.country ? `${weather.city}, ${weather.country}` : (weather.city || "-");
  document.getElementById("weather-condition").textContent = weather.weather_condition || "-";
  document.getElementById("weather-temp").textContent =
    typeof weather.temperature === "number" ? `${weather.temperature.toFixed(1)} C` : "-";
  document.getElementById("weather-humidity").textContent =
    typeof weather.humidity === "number" ? `${weather.humidity}%` : "-";
  document.getElementById("weather-wind").textContent =
    typeof weather.wind_speed === "number" ? `${weather.wind_speed} m/s` : "-";

  document.getElementById("outfit-upper").textContent = payload.upper || "-";
  document.getElementById("outfit-lower").textContent = payload.lower || "-";
  document.getElementById("outfit-footwear").textContent = payload.footwear || "-";
  document.getElementById("outfit-extra").textContent = payload.extra || "-";

  resultsEl.classList.remove("hidden");
  feedbackStatusEl.textContent = "";
  setFeedbackState(null);
}

function setFeedbackState(value) {
  feedbackGoodBtn.classList.remove("active-good");
  feedbackBadBtn.classList.remove("active-bad");

  if (value === "good") {
    feedbackGoodBtn.classList.add("active-good");
  }
  if (value === "bad") {
    feedbackBadBtn.classList.add("active-bad");
  }
}

async function fetchOutfit(city) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ city })
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || "Failed to get outfit recommendation.");
  }

  return body;
}

async function sendFeedback(feedback) {
  if (!lastPrediction) {
    feedbackStatusEl.textContent = "Get an outfit recommendation first.";
    return;
  }

  setFeedbackState(feedback);

  const payload = {
    city: lastPrediction.city,
    feedback,
    outfit: {
      upper: lastPrediction.upper,
      lower: lastPrediction.lower,
      footwear: lastPrediction.footwear,
      extra: lastPrediction.extra
    },
    weather_condition: lastPrediction.weather?.weather_condition || null,
    weather: lastPrediction.weather || null,
    features: lastPrediction.features || null
  };

  try {
    const response = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      feedbackStatusEl.textContent = "Thanks. Feedback submitted.";
      return;
    }

    feedbackStatusEl.textContent = "Feedback captured on UI (backend endpoint not available).";
  } catch {
    feedbackStatusEl.textContent = "Could not send feedback to backend.";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const city = cityInput.value.trim();
  if (!city) {
    setStatus("Please enter a city.", true);
    return;
  }

  setLoading(true);
  setStatus("Fetching weather and prediction...");

  try {
    const result = await fetchOutfit(city);
    lastPrediction = result;
    renderResult(result);
    setStatus("Recommendation ready.");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error.";
    setStatus(message, true);
    resultsEl.classList.add("hidden");
  } finally {
    setLoading(false);
  }
});

feedbackGoodBtn.addEventListener("click", () => sendFeedback("good"));
feedbackBadBtn.addEventListener("click", () => sendFeedback("bad"));
