import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="SDU Reviews — 2GIS Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        html, body, .main, .block-container {
            background-color: #f4f3ff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* ===========================
       GLOBAL COLOR PALETTE
       Primary: #827be6
       Dark:    #1c1580
       Bg:      #f4f3ff
       =========================== */

    /* Global background override */
    html, body, .main, .block-container {
        background-color: #f4f3ff !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #1c1580 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Typography */
    * {
        font-family: 'Inter', sans-serif !important;
        color: #1c1580 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ecebff !important;
        color: #1c1580 !important;
    }

    /* Sidebar text */
    .css-1wvsk6s, .css-qrbaxs, label, .stRadio label {
        color: #1c1580 !important;
    }

    /* FIX Multiselect red tags */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #827be6 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
    }

    /* FIX red text during typing */
    .stMultiSelect [data-baseweb="select"] * {
        color: #1c1580 !important;
    }

    /* Multiselect input background */
    .stMultiSelect input {
        background-color: white !important;
    }

    /* Metric Cards */
    .metric-card {
        background-color: white !important;
        padding: 18px !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(130, 123, 230, 0.15) !important;
        border-left: 4px solid #827be6 !important;
        margin-bottom: 15px !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #827be6 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    .stButton>button:hover {
        background-color: #1c1580 !important;
        transition: 0.2s;
    }

    /* Tables */
    table {
        background-color: white !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
    }
</style>
    """,
    unsafe_allow_html=True
)

def metric_card(title, value):
    st.markdown(f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
    """, unsafe_allow_html=True)

page = st.sidebar.radio("Навигация", ["Overview", "Clusters", "Topics"])

@st.cache_data
def load_data():
    df = pd.read_csv("sdu_overall_proj.csv")
    df = df.dropna(subset=["text"])
    df["clean_text"] = df["text"].str.lower().str.strip()
    return df


df = load_data()

def rating_to_sentiment(r):
    if r >= 4:
        return "positive"
    elif r <= 2:
        return "negative"
    return "neutral"

df["sentiment"] = df["rating"].apply(rating_to_sentiment)

months = {"января": "01", "февраля": "02", "марта": "03", "апреля": "04",
          "мая": "05", "июня": "06", "июля": "07", "августа": "08",
          "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"}


def parse_date(x):
    if pd.isna(x): return None
    parts = x.split()
    if len(parts) != 3: return None
    d, m, y = parts
    m = months.get(m.lower())
    return f"{y}-{m}-{d.zfill(2)}" if m else None


df["date"] = df["date"].apply(parse_date)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()


@st.cache_resource
def load_bert():
    tok = AutoTokenizer.from_pretrained("kz-transformers/kaz-roberta-conversational")
    mdl = AutoModel.from_pretrained("kz-transformers/kaz-roberta-conversational")
    mdl.eval()
    return tok, mdl


tokenizer, model = load_bert()


@st.cache_data
def embed_texts(texts):
    out = []
    for t in texts:
        inp = tokenizer(t, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            emb = model(**inp).last_hidden_state[:, 0, :].squeeze().numpy()
        out.append(emb)
    return np.vstack(out)


X = embed_texts(df["clean_text"])


st.sidebar.header("Фильтры")
years = sorted(df["year"].dropna().unique())
months_list = sorted(df["month"].dropna().unique())
sentiments = df["sentiment"].unique()

selected_years = st.sidebar.multiselect("Год", years, default=years)
selected_months = st.sidebar.multiselect("Месяц", months_list, default=months_list)
selected_sents = st.sidebar.multiselect("Сентимент", sentiments, default=sentiments)

filtered_df = df[
    df["year"].isin(selected_years)
    & df["month"].isin(selected_months)
    & df["sentiment"].isin(selected_sents)
]

# =====================
# PAGE 1 — OVERVIEW
# =====================
if page == "Overview":
    st.title("Общая статистика — 2GIS Style Dashboard")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Всего отзывов", len(filtered_df))
    with c2: metric_card("Позитивных (%)", round((filtered_df.sentiment == "positive").mean() * 100, 1))
    with c3: metric_card("Негативных (%)", round((filtered_df.sentiment == "negative").mean() * 100, 1))
    st.subheader("Распределение рейтингов")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(filtered_df["rating"], bins=20, edgecolor="black", color="#fe7070")
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(10, 5))

    st.subheader("Доля сентимента по годам") 
    yearly = filtered_df.groupby("year")["sentiment"].value_counts(normalize=True).unstack().fillna(0) 
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly.plot(kind="bar", stacked=True, ax=ax, color={'positive': '#fe7070', 'negative': '#14b2ff', 'neutral': '#ffbe51'})
    ax.set_title("Доля сентимента по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Доля")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Динамика позитивных и негативных отзывов (2020–2025)")
    timeline = filtered_df.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(16, 6))
    ax3.plot(timeline.index, timeline["positive"], label="Positive", linewidth=2, marker="o", color="#fe7070")
    ax3.plot(timeline.index, timeline["negative"], label="Negative", linewidth=2, marker="o", color="#14b2ff")
    ax3.set_title("Динамика позитивных и негативных отзывов (2020–2025)")
    ax3.set_xlabel("Время")
    ax3.set_ylabel("Количество отзывов")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig3)

    st.subheader("Распределение сентимента по месяцам (2023–2025)")
    filtered_df_period = filtered_df[(filtered_df["year_month"] >= "2023-01-01") & (filtered_df["year_month"] <= "2025-12-31")]
    monthly = filtered_df_period.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)
    monthly.index = monthly.index.strftime("%Y-%m")
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    monthly.plot(kind="bar", stacked=True, ax=ax2, color={'positive': '#39dfb6', 'negative': '#14b2ff', 'neutral': '#ffbe51'})
    ax2.set_title("Распределение сентимента по месяцам (2023–2025)", fontsize=14)
    ax2.set_xlabel("Месяц", fontsize=12)
    ax2.set_ylabel("Количество отзывов", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    df['year_month'] = df['date'].dt.to_period('M')
    # Группируем по месяцу и считаем сумму лайков
    monthly_likes = df.groupby('year_month')['reactions_total'].sum().reset_index()
    # Если хочешь в обычном текстовом формате вместо Period
    monthly_likes['year_month'] = monthly_likes['year_month'].astype(str)

    st.subheader("Сумма реакций по месяцам (топ 5)")
    top_monthly_likes = monthly_likes.sort_values('reactions_total', ascending=False).head(5)
    st.dataframe(top_monthly_likes)

    st.subheader("Просмотр отзывов по месяцу")
    selected_year_for_reviews = st.selectbox("Выберите год для просмотра отзывов", sorted(filtered_df["year"].dropna().unique()))
    selected_month_for_reviews = st.selectbox("Выберите месяц для просмотра отзывов", sorted(filtered_df["month"].dropna().unique()))
    reviews_month = filtered_df[(filtered_df["year"] == selected_year_for_reviews) & (filtered_df["month"] == selected_month_for_reviews)].sort_values('reactions_total', ascending=False)
    st.write(f"Топ 10 отзывов по реакциям за {selected_month_for_reviews}/{selected_year_for_reviews}:")
    if not reviews_month.empty:
        for row in reviews_month.head(10).itertuples():
            text_short = row.text[:300] + ('...' if len(row.text) > 300 else '')
            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 15px; margin: 10px 0; border-radius: 12px; box-shadow: 0 2px 8px rgba(130, 123, 230, 0.15); border-left: 4px solid #827be6;">
                <p style="margin: 0; color: #1c1580;"><strong>Текст:</strong> {text_short}</p>
                <p style="margin: 5px 0 0 0; color: #1c1580;"><strong>Реакции:</strong> {row.reactions_total} | <strong>Рейтинг:</strong> {row.rating}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("Нет отзывов за выбранный период.")



# =====================
# PAGE 2 — CLUSTERS
# =====================
if page == "Clusters":

    st.title("Кластеры отзывов — 2GIS Style")

    rating_scaled = (df["rating"] - 3) / 2
    X_enh = np.hstack([X, rating_scaled.values.reshape(-1, 1)])

    kmeans = KMeans(n_clusters=5, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_enh)

    st.subheader("Распределение сентимента по кластерам")

    clust = df.groupby("cluster")["sentiment"].value_counts(normalize=True).unstack().fillna(0)

    clust["satisfaction_index"] = clust.get("positive", 0) - clust.get("negative", 0)

    st.dataframe(clust)

    st.subheader("Примеры отзывов по кластеру")
    cl_sel = st.selectbox("Кластер", sorted(df["cluster"].unique()))
    for t in df[df["cluster"] == cl_sel]["clean_text"].sample(10):
        if len(t) > 400:
            t = t[:400] + "..."
        st.markdown(f"- {t}")

    st.subheader("PCA визуализация кластеров")

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_enh)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    cluster_colors = plt.cm.get_cmap("tab10", df["cluster"].nunique())

    fig, ax = plt.subplots(figsize=(12, 6))

    scatter = ax.scatter(
        df["PC1"],
        df["PC2"],
        c=df["cluster"],
        cmap=cluster_colors,
        alpha=0.7
    )

    ax.set_title("Кластеры в PCA-пространстве", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    legend = ax.legend(*scatter.legend_elements(), title="Cluster ID")
    ax.add_artist(legend)

    st.pyplot(fig)


# =====================
# PAGE 3 — TOPICS
# =====================
if page == "Topics":

    st.title("Темы отзывов — BERTopic (2GIS Style)")

    custom_stopwords = [
        "и","но","что","это","как","на","не","по","за","вы","мы","они","он","она",
        "так","такой","тут","здесь","там","если","чтобы","когда","то","у","же","ну",

        "очень","просто","нормально","просто","вообще","реально","супер","класс",'уже', 'где', 'тоже','или'

        "жақсы","өте","жоқ","екен","сол","осы","мен","сен","сіз","бір","болып",
        "әр","тағы","ғана","бар","да",'ен', "ең"

        "керемет","топ","имба","әдемі","тамаша","жаксы",

        "the","and","for","in","to","of","is","my","your","you","we","it","a",

        "университет","универ","sdu","сду","университета","университете",
        "студент","студенты","кампус","учеба"
    ]

    def remove_stopwords(text, stopwords):
        words = text.split()
        filtered = [w for w in words if w.lower() not in stopwords]
        return ' '.join(filtered)

    @st.cache_resource
    def load_topic_model(texts):
        embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
        model = BERTopic(
        language="multilingual",
        nr_topics=10        
        )
    
        embeddings = embedder.encode(texts, show_progress_bar=False)
        topics, _ = model.fit_transform(texts, embeddings)
        return model, topics

    custom_topic_names = {
        -1: 'Более позитивный', 
        0: "Смайлики",
        1: "Учебный процесс",
        2: "Портал", 
        3: "Хорошие отзывы об университете",
        4: "Столовая и еда",
        5: "Права Женщин",
        6: "Отзывы содержающие числа",
        7: "Инфраструктура и Чистота",
        8: "Религия и организации"
    }

    texts_no_stop = [remove_stopwords(t, custom_stopwords) for t in df["clean_text"]]
    topic_model, topics = load_topic_model(texts_no_stop)
    df["topic"] = topics
    topic_info = topic_model.get_topic_info()
    topic_info["Name"] = topic_info["Topic"].map(custom_topic_names)
    st.subheader("Список топиков")
    st.dataframe(topic_info[["Topic", "Count", "Name", "Representation", 'Representative_Docs']])
    # ===========================
# SENTIMENT BALANCE FOR TOPICS
# ===========================

    df["topic_name"] = df["topic"].map(custom_topic_names)

# таблица: topic_name × sentiment counts
    sentiment_counts = (
        df.groupby(['topic_name', 'sentiment'])
        .size()
        .unstack(fill_value=0)
    )

# total reviews per topic
    sentiment_counts['total'] = sentiment_counts.sum(axis=1)

# percentage of negative sentiment
    if 'negative' in sentiment_counts.columns:
        sentiment_counts['% negative'] = (
            sentiment_counts['negative'] / sentiment_counts['total'] * 100
        ).round(2)
    else:
        sentiment_counts['% negative'] = 0

    st.subheader("Сентимент по топикам (включая % негативных)")
    st.dataframe(sentiment_counts)

    topic_sentiment = (
    df.groupby("topic")["sentiment"]
      .value_counts()
      .unstack()
      .fillna(0)
      .astype(int)
      .sort_index()
    )
    colors = {
        "positive": "#61c7a3",
        "neutral": "#b7b4e3",
        "negative": "#c76262"
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    topic_sentiment.plot(
        kind="bar",
        stacked=True,
        color=[colors.get(col, "#999999") for col in topic_sentiment.columns],
        ax=ax
    )

    st.subheader("ТОП-10 отзывов по лайкам в выбранном топике")

    available_topics = sorted(df["topic"].unique())
    selected_topic = st.selectbox("Выберите Topic ID", available_topics)

    topic_reviews = df[df["topic"] == selected_topic]

    if topic_reviews.empty:
        st.write("Нет отзывов для этого топика.")
    else:
        top_reviews = topic_reviews.sort_values(by="reactions_total", ascending=False).head(10)

        for row in top_reviews.itertuples():
            text_short = row.text[:400] + ("..." if len(row.text) > 400 else "")
        
            st.markdown(f"""
            <div style="
                background-color:white;
                padding:15px;
                margin:10px 0;
                border-radius:12px;
                box-shadow:0 2px 8px rgba(130, 123, 230, 0.15);
                border-left:4px solid #827be6;
            ">
                <p style="margin:0;"><strong>Лайки:</strong> {row.reactions_total}</p>
                <p style="margin:5px 0 0 0;"><strong>Отзыв:</strong> {text_short}</p>
            </div>
            """, unsafe_allow_html=True)
        
