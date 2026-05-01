import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                         names=['item_id', 'title', 'release_date', 'video_release_date',
                                'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    return ratings, movies

@st.cache_resource
def train_models(ratings, movies):
    # SVD
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, random_state=42)
    svd.fit(trainset)

    # CB
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
    genre_matrix = movies[genre_cols].values
    cosine_sim = cosine_similarity(genre_matrix)
    movie_idx = pd.Series(movies.index, index=movies['item_id']).to_dict()

    return svd, cosine_sim, movie_idx, genre_cols

ratings, movies = load_data()
svd, cosine_sim, movie_idx, genre_cols = train_models(ratings, movies)


st.title("Movie Recommender System")

mode = st.radio("Выбери режим:",
                ["По пользователю (collaborative)",
                 "По фильму (content-based)",
                 "Новый пользователь (cold start)"])

st.divider()

if mode == "По пользователю (collaborative)":
    st.subheader("Рекомендации на основе поведения похожих пользователей")
    user_id = st.slider("Выбери ID пользователя", 1, 943, 1)
    n = st.slider("Количество рекомендаций", 5, 20, 10)

    seen = set(ratings[ratings['user_id'] == user_id]['item_id'])
    unseen = set(movies['item_id']) - seen
    predictions = [(iid, svd.predict(user_id, iid).est) for iid in unseen]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    result = []
    for item_id, score in top_n:
        title = movies[movies['item_id'] == item_id]['title'].values[0]
        result.append({'Фильм': title, 'Предсказанный рейтинг': round(score, 2)})

    st.dataframe(pd.DataFrame(result), use_container_width=True)
    st.caption(f"Пользователь уже посмотрел {len(seen)} фильмов")

elif mode == "По фильму (content-based)":
    st.subheader("Похожие фильмы по жанрам")
    movie_titles = movies['title'].tolist()
    selected = st.selectbox("Выбери фильм:", movie_titles)
    n = st.slider("Количество рекомендаций", 5, 20, 10)

    item_id = movies[movies['title'] == selected]['item_id'].values[0]
    idx = movie_idx[item_id]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]

    result = []
    for i, score in sim_scores:
        row = movies.iloc[i]
        genres = ', '.join([g for g in genre_cols if row[g] == 1])
        result.append({'Фильм': row['title'],
                       'Сходство': round(score, 3),
                       'Жанры': genres})

    st.dataframe(pd.DataFrame(result), use_container_width=True)

else:
    st.subheader("Топ фильмов для нового пользователя")
    st.info("У нового пользователя нет истории, поэтому показываем популярные фильмы с высоким рейтингом")
    min_votes = st.slider("Минимум оценок", 20, 200, 50)
    n = st.slider("Количество рекомендаций", 5, 20, 10)

    stats = ratings.groupby('item_id').agg(
        count=('rating', 'count'),
        mean_rating=('rating', 'mean')
    ).reset_index()
    popular = stats[stats['count'] >= min_votes].nlargest(n, 'mean_rating')
    popular = popular.merge(movies[['item_id', 'title']], on='item_id')

    result = [{'Фильм': r['title'],
               'Рейтинг': round(r['mean_rating'], 2),
               'Голосов': int(r['count'])}
              for _, r in popular.iterrows()]
    st.dataframe(pd.DataFrame(result), use_container_width=True)
    st.caption("После 5+ оценок система переключится на персональные рекомендации (SVD)")
