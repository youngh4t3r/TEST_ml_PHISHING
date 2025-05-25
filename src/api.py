import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from fastapi import FastAPI, Query
import pickle, pandas as pd
from feature_extraction import extract, FEATURE_NAMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
model = pickle.load(open(ROOT / "data" / "phish_model.pkl", "rb"))
THRESHOLD = 0.25

app = FastAPI()

@app.get("/")
def home():
    return {"status": "OK", "try": "/check?url=https://example.com"}

@app.get("/check")
def check_url(url: str = Query(..., description="URL для проверки")):
    feats = pd.DataFrame([extract(url)], columns=FEATURE_NAMES)
    prob = model.predict_proba(feats)[0, 1]
    return {
        "url": url,
        "phishing_probability": round(prob, 3),
        "is_phishing": bool(prob >= THRESHOLD)
    }
