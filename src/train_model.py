import pandas as pd, pathlib, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

data = pd.read_csv(DATA_DIR / "features.csv")



X = data.drop(columns="label")
y = data["label"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(Xtr, ytr)

print(classification_report(yte, model.predict(Xte)))
print("ROC-AUC:", roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))

with open(DATA_DIR / "phish_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✓ phish_model.pkl сохранён")

# -------- сохранить probs и y для экспериментов -----------
import pickle, pandas as pd, pathlib
DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

df = pd.read_csv(DATA / "features.csv")
X  = df.drop(columns=["label"])
y  = df["label"]

probs = model.predict_proba(X)[:, 1]

pickle.dump(probs, open(DATA / "probs.pkl", "wb"))
pickle.dump(y,     open(DATA / "y.pkl",     "wb"))
print("✓ probs.pkl и y.pkl сохранены")