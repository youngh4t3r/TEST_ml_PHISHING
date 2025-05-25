 🕵️‍♂️ TEST_ml_PHISHING

> Обнаружение фишинговых URL на основе признаков и логистической регрессии.  
> Почти 1 млн легитимных и 570k фишинговых ссылок, 12+ признаков, точность ~99%.

---

## 📘 Исследование по теме

Подробный текст исследования и разработки алгоритма доступен в отдельной статье:

👉 [Читать статью «Phishing Research Article»](Phishing_Research_Article.md)


## 🚀 Возможности

- Извлечение признаков из URL
- Обучение модели (логистическая регрессия)
- Подбор оптимального порога
- CLI-интерфейс для проверки URL

---

## 📦 Установка

```bash
git clone https://github.com/youngh4t3r/TEST_ml_PHISHING.git
cd TEST_ml_PHISHING
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

---

## 🔍 Проверка URL

```bash
python src\predict.py https://paypal.com
python src\predict.py http://login-secure-update.ru
```

---

## 📁 Структура

```
src/
├─ extract_features.py
├─ feature_extraction.py
├─ train_model.py
├─ predict.py
├─ exp_threshold.py
```

---

## 📈 Метрики (Logistic Regression)

| Метрика           | Значение |
|-------------------|----------|
| Accuracy          | 0.99     |
| Recall (фишинг)   | 0.99     |
| Precision (фишинг)| 1.00     |
| ROC-AUC           | 0.998    |

---

## 📝 Лицензия

MIT (см. [LICENSE](LICENSE))

---

## 💬 Автор

@youngh4t3r  
Проект в рамках развития дипломного проекта.

---

## 💡 Идеи на будущее

- Перевод модели на RandomForest или XGBoost
- Добавление FastAPI / Telegram-бота
- Docker-обёртка
- Визуализация метрик (PR-кривая)
