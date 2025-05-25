# TEST_ml_PHISHING

Proof-of-concept: выявление фишинговых URL с помощью простых URL-признаков  
и базовых моделей машинного обучения (MVP).

## Шаги
1. `python -m venv .venv && .\.venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Поместите датасеты в `data/`, запустите `src/extract_features.py`
4. `python src/train_model.py` — получаем `phish_model.pkl`
