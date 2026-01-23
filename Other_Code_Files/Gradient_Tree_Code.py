#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
try:
    data = pd.read_csv('Raw data chlorocline.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'Raw data chlorocline.csv' not found.")
    exit(1)

# Feature and target columns
feature_cols = ['shallow_depth_s', 'Deep_depth_s', 'Deep_depth_t']
target_col = 'chlorophyll_depth'

def remove_outliers(df):
    """
    Removes rows containing outliers in the specified columns using the Z-score method.
    """
    df_cleaned = df.copy()
    cols_to_check = feature_cols + [target_col]
    present_cols = [c for c in cols_to_check if c in df_cleaned.columns]
    
    for col in present_cols:
        if df_cleaned[col].dtype == object:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        if df_cleaned[col].dropna().nunique() <= 1:
            continue
        try:
            z = stats.zscore(df_cleaned[col].astype(float))
        except Exception:
            continue
        outlier_mask = np.abs(z) > 2
        df_cleaned = df_cleaned[~outlier_mask]

    return df_cleaned

def coerce_and_clean_numeric(df, cols, target_col):
    """Coerce listed columns to numeric, replace infs, drop rows with NaNs."""
    df2 = df.copy()
    
    for c in cols + [target_col]:
        if c in df2.columns:
            coerced = pd.to_numeric(df2[c], errors='coerce')
            df2[c] = coerced

    # Replace inf values with NaN
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    
    present_cols = [c for c in cols + [target_col] if c in df2.columns]
    df2_kept = df2.dropna(subset=present_cols) if present_cols else df2.copy()
    
    return df2_kept

# Apply cleaning and outlier removal
data_numeric = coerce_and_clean_numeric(data, feature_cols, target_col)
data_cleaned = remove_outliers(data_numeric)

print(f"Original shape: {data.shape}")
print(f"Cleaned shape: {data_cleaned.shape}")

# Prepare data for training
if target_col not in data_cleaned.columns:
    print(f"Error: Target column '{target_col}' not found in cleaned data.")
    exit(1)

X = data_cleaned[feature_cols]
y = data_cleaned[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Gradient Boosting Regressor
# You can tune hyperparameters like n_estimators, learning_rate, max_depth, etc.
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Predict
y_pred = gb_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Gradient Boosted Tree RMSE: {rmse}")

# Scatterplot of Model Predictions vs Actual Depths
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Chlorophyll Depth')
plt.ylabel('Predicted Chlorophyll Depth')
plt.title('Gradient Boosted Tree: Actual vs Predicted Chlorophyll Depth')
plt.legend()
plt.grid(True)
plt.savefig('gradient_tree_prediction_scatter.png')
print("Scatterplot saved as 'gradient_tree_prediction_scatter.png'")
plt.show()
