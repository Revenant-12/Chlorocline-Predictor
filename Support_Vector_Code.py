#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import os

def main():
    # Load data
    filename = 'Raw data chlorocline.csv'
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found.")
        return

    print(f"Loading {filename}...")
    data = pd.read_csv(filename, encoding='latin-1')

    # Feature and target columns
    feature_cols = ['shallow_depth_s', 'deep_depth_s', 'Deep_depth_t']
    target_col = 'chlorophyll_depth'

    # --- Preprocessing Functions ---

    def coerce_and_clean_numeric(df, cols, target_col):
        """Coerce listed columns to numeric, replace infs, drop rows with NaNs."""
        df2 = df.copy()
        
        # Only coerce the numeric feature/target columns
        for c in cols + [target_col]:
            if c in df2.columns:
                # attempt coercion
                coerced = pd.to_numeric(df2[c], errors='coerce')
                df2[c] = coerced

        # Replace inf values with NaN
        df2 = df2.replace([np.inf, -np.inf], np.nan)

        present_cols = [c for c in cols + [target_col] if c in df2.columns]
        
        before = df2.shape[0]
        
        # Drop rows with NaNs in the relevant columns
        df2_kept = df2.dropna(subset=present_cols) if present_cols else df2.copy()
        after = df2_kept.shape[0]

        print(f"Dropped {before - after} rows with NaNs in required columns (remaining {after} rows).")

        return df2_kept

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
            outlier_mask = np.abs(z) > 2 # standard threshold
            df_cleaned = df_cleaned[~outlier_mask]

        return df_cleaned

    # --- Apply Preprocessing ---
    print("--- Data Preprocessing ---")
    data_numeric = coerce_and_clean_numeric(data, feature_cols, target_col)
    data_cleaned = remove_outliers(data_numeric)
    print(f"Original shape: {data.shape}")
    print(f"Final data shape: {data_cleaned.shape}")

    # --- Support Vector Machine ---
    print("\n--- Training Support Vector Machine ---")

    X = data_cleaned[feature_cols]
    y = data_cleaned[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features (Critical for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and fit SVR
    # Using RBF kernel by default as it is versatile
    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train)

    # Predict
    y_pred = svr.predict(X_test_scaled)

    # Calculate MSE and RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)

    print(f"\nSupport Vector Machine Results:")
    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")

    # Plotting Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Chlorophyll Depth')
    plt.ylabel('Predicted Chlorophyll Depth')
    plt.title('SVR: Actual vs Predicted Chlorophyll Depth')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()