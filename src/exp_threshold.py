import pathlib, pickle
from sklearn.metrics import precision_recall_curve
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

probs = pickle.load(open(DATA / "probs.pkl", "rb"))
y     = pickle.load(open(DATA / "y.pkl",     "rb"))

prec, rec, thr = precision_recall_curve(y, probs)
thr = np.append(thr, 1)        # делаем массив такой же длины, как prec/rec

print(" threshold   recall  precision")
print("--------------------------------")
for t, r, p in zip(thr, rec, prec):
    if r >= 0.90:              # печатаем ВСЕ пороги с recall >= 0.90
        print(f"{t:9.2f}   {r:6.2f}    {p:8.2f}")

import pandas as pd

table = pd.DataFrame({"threshold": thr, "recall": rec, "precision": prec})
table[table.recall >= 0.90].to_csv("threshold_table.csv", index=False)
print("✓ threshold_table.csv сохранён")
