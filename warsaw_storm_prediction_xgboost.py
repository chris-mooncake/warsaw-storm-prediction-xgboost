"""Storm prediction model using Open-Meteo and NASA POWER weather data for Warsaw."""

import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, plot_importance


# =========================
# Download and Save/Load
# =========================

def validate_nasa_power(df):
    """Validate that NASA POWER dataset starts no earlier than 2000."""
    if "date" not in df.columns:
        return False
    min_date = pd.to_datetime(df["date"]).min()
    return min_date >= pd.Timestamp("2000-01-01")


def load_or_download(path, download_fn, validate_fn=None):
    """Load cached data or download and validate it if not found."""
    if os.path.exists(path):
        print(f"[✔] Found cached data: {path}")
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        elif "index" in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"]).dt.date

        if validate_fn and not validate_fn(df):
            print(f"❌ Cached file {path} failed validation. Re-downloading...")
            os.remove(path)
            return load_or_download(path, download_fn, validate_fn)

        return df

    print(f"[↓] Downloading data to: {path}")
    df = download_fn()

    if validate_fn and not validate_fn(df):
        raise ValueError(f"❌ Downloaded data failed validation: {path}")

    df.to_csv(path, index=False)
    print(f"[💾] Saved valid data to: {path}")
    return df


def download_open_meteo():
    """Download weather data from Open-Meteo archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.23,
        "longitude": 21.01,
        "start_date": "2010-01-01",
        "end_date": "2024-12-31",
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max"
        ]),
        "timezone": "Europe/Warsaw"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["daily"]
    df = pd.DataFrame(data)
    df.rename(columns={"time": "date"}, inplace=True)
    return df


def download_nasa_power():
    """Download weather data from NASA POWER API."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": "20100101",
        "end": "20241231",
        "latitude": 52.23,
        "longitude": 21.01,
        "parameters": ",".join(["RH2M", "WS2M", "PRECTOTCORR", "T2M", "PS"]),
        "format": "JSON",
        "community": "AG"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    raw_params = data["properties"]["parameter"]
    dates = list(raw_params[list(raw_params.keys())[0]].keys())
    df = pd.DataFrame({"date": dates})
    for param, values in raw_params.items():
        df[param] = pd.Series(values).values
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# =========================
# Data Cleaning
# =========================

def clean_and_merge_data(df1, df2):
    """Clean and merge Open-Meteo and NASA POWER datasets."""
    df1["date"] = pd.to_datetime(df1["date"])
    df2["date"] = pd.to_datetime(df2["date"])
    df = pd.merge(df1, df2, on="date", how="inner")
    print(f"\n[🔗] Merged dataset shape: {df.shape}")
    if df.empty:
        raise ValueError("❌ Merged dataset is empty!")

    imputer = KNNImputer(n_neighbors=5)
    feature_cols = df.columns.difference(["date"])
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df


# =========================
# Feature Engineering
# =========================

def engineer_features(df):
    """Create target and time-based features, and add rolling stats."""
    df["storm"] = ((df["precipitation_sum"] > 10) |
                   (df["wind_speed_10m_max"] > 50)).astype(int)
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df = df.sort_values("date")
    rolling_cols = [
        "temperature_2m_max", "temperature_2m_min",
        "wind_speed_10m_max", "precipitation_sum"
    ]
    for col in rolling_cols:
        df[f"{col}_roll3"] = df[col].rolling(window=3, min_periods=1).mean()
        df[f"{col}_roll7"] = df[col].rolling(window=7, min_periods=1).mean()
    return df


# =========================
# Validation
# =========================

def validate_binary_target(series, name="target"):
    """Check if the target variable contains only binary values."""
    unique_values = set(series.unique())
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"❌ {name} must contain only binary values (0 or 1). Found: {unique_values}")
    print(f"[✔] {name} validated as binary: {unique_values}")


# =========================
# Model Building
# =========================

def build_model():
    """Initialize an XGBoost classifier."""
    return XGBClassifier(eval_metric="logloss")


# =========================
# Model Training
# =========================

def train_model(model, x_train, y_train):
    """Train XGBoost with grid search and return best estimator."""
    param_grid = {
        'learning_rate': [0.1, 0.2, 0.3],
        'max_depth': [5, 7, 9],
        'n_estimators': [100, 200]
    }
    scorer = make_scorer(
        lambda y_true, y_pred: precision_recall_fscore_support(
            y_true, y_pred, average='binary')[2]
    )
    grid_search = GridSearchCV(
        model, param_grid, scoring=scorer, cv=3, n_jobs=-1
    )
    grid_search.fit(x_train, y_train)
    print("\n✅ Best Parameters:")
    print(grid_search.best_params_)
    return grid_search.best_estimator_


# =========================
# Model Evaluation
# =========================

def evaluate_model(model, x_train, y_train, x_test, y_test):
    """Evaluate model performance with reports, thresholds, and plots."""
    joblib.dump(model, "storm_predictor.pkl")
    print("[💾] Model saved to: storm_predictor.pkl")

    y_pred_train = model.predict(x_train)
    print("\n📊 EVALUATION ON TRAINING DATA:")
    print(classification_report(y_train, y_pred_train))
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train)
    plt.title("Train Confusion Matrix")
    plt.show()

    print("\n=================== TEST ON UNSEEN 2023-2024 DATA ===================")
    probs = model.predict_proba(x_test)[:, 1]
    thresholds = [i / 100 for i in range(10, 91, 5)]

    metrics = []
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='binary')
        metrics.append((threshold, precision, recall, f1))

    df_metrics = pd.DataFrame(metrics, columns=["Threshold", "Precision", "Recall", "F1"])
    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics["Threshold"], df_metrics["Precision"], label="Precision")
    plt.plot(df_metrics["Threshold"], df_metrics["Recall"], label="Recall")
    plt.plot(df_metrics["Threshold"], df_metrics["F1"], label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Storm Detection Metrics vs. Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_threshold = df_metrics.sort_values("F1", ascending=False).iloc[0]["Threshold"]
    print(f"📌 Best threshold by F1 score: {best_threshold:.2f}")

    final_preds = (probs > best_threshold).astype(int)
    print("\n📊 EVALUATION USING OPTIMAL THRESHOLD:")
    print(classification_report(y_test, final_preds))
    ConfusionMatrixDisplay.from_predictions(y_test, final_preds)
    plt.title(f"Test Confusion Matrix (Threshold={best_threshold:.2f})")
    plt.show()

    plot_importance(model, max_num_features=10)
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()


# =========================
# Main Entry Point
# =========================

def main():
    """Main training and evaluation pipeline."""
    df_meteo = load_or_download("open_meteo_daily.csv", download_open_meteo)
    df_nasa = load_or_download("nasa_power_daily.csv", download_nasa_power, validate_nasa_power)
    df_clean = clean_and_merge_data(df_meteo, df_nasa)
    df_features = engineer_features(df_clean)

    df_train = df_features[df_features["date"] < datetime(2023, 1, 1)]
    df_test = df_features[df_features["date"] >= datetime(2023, 1, 1)]

    excluded_features = ["date", "storm", "precipitation_sum", "wind_speed_10m_max"]
    features = df_features.columns.difference(excluded_features)

    x_train, y_train = df_train[features], df_train["storm"]
    x_test, y_test = df_test[features], df_test["storm"]

    validate_binary_target(y_train, name="y_train")
    validate_binary_target(y_test, name="y_test")

    print("\n⚠️ Class distribution BEFORE SMOTE (train):")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)
    x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

    print("\n✅ Class distribution AFTER SMOTE:")
    print(pd.Series(y_train_bal).value_counts())

    model = build_model()
    trained_model = train_model(model, x_train_bal, y_train_bal)
    evaluate_model(trained_model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
