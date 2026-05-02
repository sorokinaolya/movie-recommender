# movie-recommender
Рекомендательная система фильмов на основе SVD и контентной фильтрации. MovieLens 100k

# Movie Recommender System

Рекомендательная система фильмов, реализующая два подхода к персонализации и решение проблемы холодного старта.

## Демо

[Открыть приложение на Hugging Face](https://huggingface.co/spaces/olynchik/movie-recommender)

## Описание

Проект построен на датасете MovieLens 100k (100 000 оценок, 943 пользователя, 1682 фильма).

**Collaborative Filtering (SVD)** - матричная факторизация на основе истории оценок пользователей. RMSE = 0.935, MAE = 0.737 на кросс-валидации (5 фолдов). Precision@10 и Recall@10 считаются на отложенной выборке (20%).

**Content-Based Filtering** - косинусное сходство по бинарным жанровым векторам (18 жанров). Не требует истории пользователя.

**Cold Start** - popularity baseline для новых пользователей: топ фильмов по среднему рейтингу с порогом минимум 50 оценок. После 5+ оценок система переключается на SVD.

## Стек

Python, Streamlit, scikit-surprise, scikit-learn, pandas, numpy

## Структура

| Файл | Описание |
|---|---|
| `rec_proj_movie.ipynb` | анализ, обучение моделей, метрики |
| `app.py` | streamlit приложение |

## Запуск локально

```bash
pip install streamlit pandas numpy scikit-learn scikit-surprise
streamlit run app.py
```
