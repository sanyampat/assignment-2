from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

def preprocess_data(df, target_column, feature_columns):

    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Fill missing values
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

    return X, y


@app.route("/")
def home():
    return {"message": "Naive Bayes API is running"}


@app.route("/train", methods=["POST"])
def train_model():

    try:

        file = request.files["file"]
        target_column = request.form["target"]
        test_size = float(request.form.get("test_size", 0.2))
        random_state = int(request.form.get("random_state", 42))

        df = pd.read_csv(file)

        feature_columns = [col for col in df.columns if col != target_column]

        X, y = preprocess_data(df, target_column, feature_columns)

        if y.nunique() > 15:
            return jsonify({"error": "Target appears continuous. Classification required."})

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)

        return jsonify({
            "accuracy": accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "confusion_matrix": cm,
            "classification_report": report
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)