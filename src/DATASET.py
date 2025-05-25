"""
Склейка разных источников → urls_labeled.csv
  • phishing  (label 1)
  • legitimate(label 0)
Запуск: python src/DATASET.py
"""

import pandas as pd, pathlib, re
from urllib.parse import urlparse

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT  = DATA / "urls_labeled.csv"

# ------------------------------------------------------------------ helpers
def is_valid(url: str) -> bool:
    """Отбрасываем пустые, без домена, без точки и mailto:javascript:…"""
    if not isinstance(url, str) or url.startswith(("mailto:", "javascript:")):
        return False
    if "://" not in url:
        url = "http://" + url
    try:
        host = urlparse(url).netloc
        return bool(host) and "." in host
    except Exception:
        return False


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем 1 колонку url, валидируем, приводим к str, убираем дубликаты"""
    df = df.astype(str, errors="ignore")
    df = df.rename(columns=lambda c: c.lower())
    df = df[["url"]]
    df["url"] = df["url"].str.strip()
    df = df[df["url"].apply(is_valid)]
    return df.drop_duplicates()

# ------------------------------------------------------------------ 1. phishing (label=1)
phish_frames = []

# 1.1 verified_online.csv
phish_frames.append(pd.read_csv(DATA / "verified_online.csv", usecols=["url"]))

# 1.2 dataset_phishing.csv
phish_frames.append(pd.read_csv(DATA / "dataset_phishing.csv").rename(columns=str.lower)[["url"]])

# 1.3 phishing_site_urls.csv
phish_frames.append(pd.read_csv(DATA / "phishing_site_urls.csv").rename(columns=str.lower)[["url"]])

# 1.4 feed.txt (plain list)
phish_frames.append(pd.read_csv(DATA / "feed.txt", names=["url"]))

phish = pd.concat(phish_frames, ignore_index=True)
phish = clean_df(phish)
phish["label"] = 1
print("✓ phishing rows:", len(phish))

# ------------------------------------------------------------------ 2. legitimate (label=0)
legit_frames = []

# Majestic Million: берём 1M топ-доменов
majestic = pd.read_csv(
    DATA / "majestic_million.csv",
    usecols=["Domain"],           # capital 'D'
    dtype=str
).rename(columns={"Domain": "domain"})
majestic["url"] = "https://" + majestic["domain"]
legit_frames.append(majestic[["url"]])

legit = pd.concat(legit_frames, ignore_index=True)
legit = clean_df(legit)
legit["label"] = 0
print("✓ legit rows:", len(legit))

# ------------------------------------------------------------------ 3. объединяем
dataset = pd.concat([phish, legit], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42)      # случайно перемешать
dataset.to_csv(OUT, index=False)
print("⇒ saved:", OUT.relative_to(DATA), "rows:", len(dataset))
