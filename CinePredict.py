import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, chi2_contingency
import ast

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="CinePredict | Movie Success Predictor",
    page_icon="🎬",
    layout="wide"
)

plt.style.use("default")

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #FF4B4B;
}
.sub-title {
    font-size: 18px;
    color: #555;
}
.metric-box {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="main-title">🎬 CinePredict Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict Movie Success using Machine Learning</div>', unsafe_allow_html=True)
st.divider()

# ---------------- Sidebar ----------------
st.sidebar.title("🎥 CinePredict")
st.sidebar.markdown("Upload data & explore insights")

with st.sidebar.expander("ℹ️ About CinePredict", expanded=True):
    st.markdown("""
    **CinePredict predicts movie success based on:**
    - Budget  
    - Popularity  
    - Runtime  
    - Vote Average  

    Includes:
    - EDA  
    - Statistical Tests  
    - ML Prediction
    """)

uploaded_file = st.sidebar.file_uploader("📂 Upload Movies CSV", type=["csv"])

# ---------------- Load Data ----------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df = df[["budget", "revenue", "popularity", "runtime",
             "vote_average", "title", "genres"]]

    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    df.dropna(inplace=True)

    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    def extract_genre(x):
        try:
            g = ast.literal_eval(x)
            return g[0]['name'] if g else "Unknown"
        except:
            return "Unknown"

    df["main_genre"] = df["genres"].apply(extract_genre)

    # ---------------- Sidebar Filters ----------------
    st.sidebar.subheader("🔍 Filters")
    selected_genres = st.sidebar.multiselect(
        "Genre",
        options=df["main_genre"].unique(),
        default=df["main_genre"].unique()
    )
    min_votes = st.sidebar.slider("Minimum Vote Average", 0.0, 10.0, 5.0)

    filtered_df = df[
        (df["main_genre"].isin(selected_genres)) &
        (df["vote_average"] >= min_votes)
    ]

    # ---------------- KPI Cards ----------------
    col1, col2, col3 = st.columns(3)
    col1.metric("🎞 Total Movies", len(filtered_df))
    col2.metric("✅ Success Rate", f"{filtered_df['success'].mean()*100:.1f}%")
    col3.metric("🎬 Genres", filtered_df["main_genre"].nunique())

    st.divider()

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Exploratory Analysis",
        "📈 Statistics",
        "🤖 ML Model",
        "🎬 Prediction"
    ])

    # ---------------- TAB 1: EDA ----------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(filtered_df.head(), use_container_width=True)

        st.subheader("Budget vs Revenue")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=filtered_df,
            x="budget",
            y="revenue",
            hue="success",
            palette="coolwarm",
            ax=ax
        )
        st.pyplot(fig, use_container_width=False)

    # ---------------- TAB 2: Statistics ----------------
    with tab2:
        st.subheader("Descriptive Statistics")
        st.dataframe(filtered_df.describe(), use_container_width=True)

        st.subheader("Statistical Tests")

        t_stat, p_val = ttest_ind(
            filtered_df[filtered_df["success"] == 1]["vote_average"],
            filtered_df[filtered_df["success"] == 0]["vote_average"]
        )
        st.success(f"T-Test (Vote Avg vs Success): t = {t_stat:.2f}, p = {p_val:.4f}")

        contingency = pd.crosstab(filtered_df["main_genre"], filtered_df["success"])
        chi2, p, dof, exp = chi2_contingency(contingency)
        st.info(f"Chi-Square (Genre vs Success): χ² = {chi2:.2f}, p = {p:.4f}")

    # ---------------- TAB 3: ML Model ----------------
    with tab3:
        st.subheader("Random Forest Classifier")

        features = filtered_df[["budget", "popularity", "runtime", "vote_average"]]
        target = filtered_df["success"]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig, use_container_width=False)

    # ---------------- TAB 4: Prediction ----------------
    with tab4:
        st.subheader("Predict Movie Success")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                budget = st.number_input("Budget ($)", 1000, 500000000, 10000000, step=1000000)
                popularity = st.slider("Popularity", 0.0, 100.0, 10.0)

            with col2:
                runtime = st.slider("Runtime (mins)", 30, 300, 120)
                vote_avg = st.slider("Vote Average", 0.0, 10.0, 6.5)

            predict = st.form_submit_button("🎯 Predict")

        if predict:
            input_df = pd.DataFrame([{
                "budget": budget,
                "popularity": popularity,
                "runtime": runtime,
                "vote_average": vote_avg
            }])

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][pred]

            if pred == 1:
                st.success(f"🌟 Successful Movie ({prob:.2%} confidence)")
            else:
                st.error(f"🚫 Unsuccessful Movie ({prob:.2%} confidence)")

    # ---------------- Download ----------------
    st.divider()
    st.download_button(
        "⬇ Download Filtered Dataset",
        filtered_df.to_csv(index=False),
        "filtered_movies.csv",
        "text/csv"
    )

else:
    st.info("👈 Upload a movie dataset to begin")
