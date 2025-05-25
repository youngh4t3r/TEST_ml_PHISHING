"""
Логистическая регрессия с class_weight="balanced"
и сравнение с текущей моделью
Запуск: python src/exp_reweight.py
"""

import pathlib, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

Xy = pd.read_csv(DATA / "features.csv")
X   = Xy.drop(columns=["label"])
y   = Xy["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

#   новая модель
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(Xtr, ytr)

y_pred = model.predict(Xte)
print(classification_report(yte, y_pred))
print("ROC-AUC:", roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
