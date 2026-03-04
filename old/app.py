import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Naive Bayes Classification Dashboard",
    layout="wide"
)

st.title("Naive Bayes Classification Dashboard")
st.markdown("Train and evaluate a Naive Bayes classifier on structured datasets.")

# Sidebar
st.sidebar.header("Configuration")

data_option = st.sidebar.radio(
    "Data Source",
    ["Upload CSV", "Use Credit.csv"]
)

df = None

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("Credit.csv")
    except:
        st.sidebar.error("Credit.csv not found.")

# If dataset loaded
if df is not None:

    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Dataset Overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head(), use_container_width=True)

    # Target Selection
    target_column = st.sidebar.selectbox("Target Column", df.columns)

    if target_column:

        y = df[target_column]

        # Automatic Classification Detection
        if y.dtype == "object" or y.nunique() <= 15:
            st.success("Detected task type: Classification")
        else:
            st.error("Target appears continuous. Naive Bayes requires classification.")
            st.stop()

        # Show Class Distribution
        st.subheader("Target Distribution")
        fig_dist, ax_dist = plt.subplots()
        y.value_counts().plot(kind="bar", ax=ax_dist)
        ax_dist.set_xlabel("Class")
        ax_dist.set_ylabel("Count")
        st.pyplot(fig_dist)

        feature_columns = st.sidebar.multiselect(
            "Feature Columns",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )

        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.sidebar.number_input("Random State", value=42, step=1)

        if st.sidebar.button("Train Model"):

            with st.spinner("Training model..."):

                try:
                    X = df[feature_columns].copy()
                    y = df[target_column].copy()

                    # Handle missing values
                    X = X.fillna(X.mode().iloc[0])

                    # Encode categorical features
                    for col in X.columns:
                        if X[col].dtype == "object":
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))

                    # Encode target
                    if y.dtype == "object":
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=int(random_state)
                    )

                    model = GaussianNB()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    # Performance Metrics
                    st.subheader("Model Performance")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy", f"{accuracy:.4f}")
                    m2.metric("Training Samples", len(X_train))
                    m3.metric("Testing Samples", len(X_test))

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")

                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)

                    # Classification Report
                    st.subheader("Detailed Report")
                    st.code(classification_report(y_test, y_pred))

                except Exception as e:
                    st.error(f"Model error: {e}")