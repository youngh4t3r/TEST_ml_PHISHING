import pandas as pd, pathlib
from tqdm import tqdm
from feature_extraction import extract, FEATURE_NAMES

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"                      
RAW = ROOT / "data" / "urls_labeled.csv"   # подготовленный датасет (url,label)

df = pd.read_csv(RAW)

records = [extract(u) for u in tqdm(df["url"], desc="Extracting")]
features = pd.DataFrame(records, columns=FEATURE_NAMES)
features["label"] = df["label"]

features.to_csv(DATA_DIR / "features.csv", index=False)
print("✓ features.csv сохранён. Строк:", len(features))
