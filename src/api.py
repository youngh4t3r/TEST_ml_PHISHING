import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))  # добавляем src/ в путь

from fastapi import FastAPI, Query
import pickle
import pandas as pd
from feature_extraction import extract, FEATURE_NAMES

# Инициализация
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
model = pickle.load(open(DATA / "phish_model.pkl", "rb"))
THRESHOLD = 0.25  # подстрой под свой

app = FastAPI(title="Phishing URL Detection API")

# Главная страница
@app.get("/")
def home():
    return {"status": "OK", "example": "/check?url=https://example.com"}

# Основной маршрут для проверки URL
@app.get("/check")
def check_url(url: str = Query(..., description="URL для проверки")):
    try:
        feats = pd.DataFrame([extract(url)], columns=FEATURE_NAMES)
        prob = float(model.predict_proba(feats)[0, 1])
        return {
            "url": url,
            "phishing_probability": round(prob, 3),
            "is_phishing": prob >= THRESHOLD
        }
    except Exception as e:
        return {"error": str(e), "url": url}
