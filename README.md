
# Storm Prediction Model - Warsaw

## 📘 Objective

Develop a machine learning pipeline to predict **stormy weather conditions** using historical meteorological data from **multiple external APIs**, applying **feature engineering, KNN imputation**, and **XGBoost classification**, with model evaluation based on binary classification metrics.

---

## 🗂️ Project Structure

### 1. Data Cleaning
- **Sources**:
  - Open-Meteo API (`temperature`, `precipitation`, `wind`)
  - NASA POWER API (`humidity`, `pressure`, `temperature`, `wind speed`)
- **Logic**:
  - Data loaded using a reusable wrapper (`load_or_download`)
  - File caching and validation handled
  - Datasets merged on the `date` key
  - Imputation using `KNNImputer`

### 2. Feature Engineering
- Target variable: `storm` defined as:
  - `precipitation_sum > 10 mm` OR `wind_speed_10m_max > 50 km/h`
- Time-based features:
  - `month`, `dayofyear`
- Rolling features added:
  - 3-day and 7-day averages for each meteorological series

### 3. Model Build
- Uses `XGBoostClassifier` with:
  - Log loss as the objective metric
  - GridSearchCV for hyperparameter tuning

### 4. Model Training
- Training dataset is resampled using `SMOTE` to correct class imbalance
- GridSearch optimization of:
  - `learning_rate`
  - `max_depth`
  - `n_estimators`
- Best estimator returned and stored with `joblib`

### 5. Model Evaluation
- Reports:
  - Classification Report
  - Confusion Matrix (train & test)
- Threshold Optimization:
  - Calculates precision, recall, F1 for thresholds from `0.1` to `0.9`
  - Selects best threshold by maximum F1
- Feature importance plotted

---

## 📥 Inputs

### Open-Meteo API  
- **URL**: `https://archive-api.open-meteo.com/v1/archive`
- **Features**:
  - temperature_2m_max
  - temperature_2m_min
  - precipitation_sum
  - wind_speed_10m_max

### NASA POWER API  
- **URL**: `https://power.larc.nasa.gov/api/temporal/daily/point`
- **Features**:
  - T2M, RH2M, WS2M, PS, PRECTOTCORR
 
### Data Sources

This project uses publicly available weather data from the following sources:

- [Open-Meteo Archive API](https://open-meteo.com/en/docs#historical-api)  
  - Free weather data API with historical and forecast data.
  - [Terms of Service](https://open-meteo.com/en/terms)

- [NASA POWER API](https://power.larc.nasa.gov/docs/services/api/)  
  - Provides satellite-based weather and solar data for research and agriculture.
  - [Terms of Use](https://power.larc.nasa.gov/docs/services/api/#terms-of-use)


---

## 📤 Outputs

- Trained model: `storm_predictor.pkl`
- Plots:
  - Confusion matrices
  - Metric vs. threshold
  - Feature importances

---

## 🧪 Validation

- Function `validate_binary_target` checks that `storm` contains only 0 and 1

---

## ⚙️ Dependencies

```bash
pip install pandas requests matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

---

## 📊 Evaluation Metrics

| Metric        | Description                               |
|---------------|-------------------------------------------|
| Precision     | Correct positive predictions              |
| Recall        | Actual positives retrieved                |
| F1-score      | Harmonic mean of precision and recall     |
| Accuracy      | Overall correctness                       |
| Confusion Matrix | Visual prediction breakdown            |

---

## 📈 Example Output Summary

```
[✔] Found cached data: open_meteo_daily.csv
[✔] Found cached data: nasa_power_daily.csv

[🔗] Merged dataset shape: (5479, 10)
[✔] y_train validated as binary: {np.int64(0), np.int64(1)}
[✔] y_test validated as binary: {np.int64(0), np.int64(1)}

⚠️ Class distribution BEFORE SMOTE (train):
storm
0    4541
1     207
Name: count, dtype: int64

✅ Class distribution AFTER SMOTE:
storm
0    4541
1    4541
Name: count, dtype: int64

✅ Best Parameters:
{'learning_rate': 0.3, 'max_depth': 7, 'n_estimators': 200}
[💾] Model saved to: storm_predictor.pkl

📊 EVALUATION ON TRAINING DATA:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      4541
           1       1.00      1.00      1.00       207

    accuracy                           1.00      4748
   macro avg       1.00      1.00      1.00      4748
weighted avg       1.00      1.00      1.00      4748


=================== TEST ON UNSEEN 2023-2024 DATA ===================
📌 Best threshold by F1 score: 0.90

📊 EVALUATION USING OPTIMAL THRESHOLD:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       706
           1       0.72      0.72      0.72        25

    accuracy                           0.98       731
   macro avg       0.86      0.86      0.86       731
weighted avg       0.98      0.98      0.98       731
```

---

## 🧹 Code Quality & Compliance

- ✅ **Pylint score**: 9.88/10  
- ✅ **PEP8 compliant**
- ✅ All imports grouped and ordered
- ✅ Each function has a docstring
- ✅ API requests use `timeout` argument
- ✅ Snake case for all variables
- ✅ Line length limited to 100 characters
