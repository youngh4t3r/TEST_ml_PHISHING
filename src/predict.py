import sys, pathlib, pickle, pandas as pd
from feature_extraction import extract, FEATURE_NAMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
model = pickle.load(open(ROOT / "data" / "phish_model.pkl", "rb"))

THRESHOLD = 0.25  # ← здесь ставишь свой подобранный порог

def predict_url(url: str):
    feats = pd.DataFrame([extract(url)], columns=FEATURE_NAMES)
    prob = model.predict_proba(feats)[0, 1]
    return {
        "url": url,
        "phishing_probability": round(prob, 3),
        "is_phishing": bool(prob >= THRESHOLD)
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py <URL>")
    else:
        result = predict_url(sys.argv[1])
        print(result)
