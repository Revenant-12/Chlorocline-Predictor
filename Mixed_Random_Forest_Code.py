#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from merf import MERF
import os

def main():
    # Load data
    filename = 'Raw data chlorocline.csv'
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found.")
        return

    print(f"Loading {filename}...")
    try:
        data = pd.read_csv(filename, encoding='latin-1')
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Feature and target columns
    feature_cols = ['shallow_depth_s', 'deep_depth_s', 'Deep_depth_t']
    target_col = 'chlorophyll_depth'
    group_col = 'Site'

    # --- Preprocessing Functions ---

    def coerce_and_clean_numeric(df, cols, target_col, group_col):
        """Coerce listed columns to numeric, replace infs, drop rows with NaNs, and provide diagnostics. 
        
        Returns the cleaned DataFrame (rows kept). Also saves dropped rows to 'dropped_rows_inspection.csv'.
        """
        df2 = df.copy()
        converted = {}
        coercion_examples = {}

        # Only coerce the numeric feature/target columns
        for c in cols + [target_col]:
            if c in df2.columns:
                before_na = df2[c].isna().sum()
                # attempt coercion and capture examples of values that fail
                coerced = pd.to_numeric(df2[c], errors='coerce')
                coerced_mask = df2[c].notna() & coerced.isna()
                coercion_examples[c] = df2.loc[coerced_mask, c].head(5).tolist()
                df2[c] = coerced
                after_na = df2[c].isna().sum()
                converted[c] = after_na - before_na

        # Replace inf values with NaN
        df2 = df2.replace([np.inf, -np.inf], np.nan)

        print("Coercion summary (new NaNs introduced by coercion):")
        for k, v in converted.items():
            print(f"  {k}: +{v} NaNs")

        # Check required columns
        cols_to_check = cols + [target_col]
        if group_col in df2.columns:
            cols_to_check.append(group_col)
        else:
            print(f"Warning: Group column '{group_col}' not found in DataFrame.")
        
        present_cols = [c for c in cols_to_check if c in df2.columns]
        
        before = df2.shape[0]
        
        # Drop rows with NaNs in the relevant columns
        dropped_rows = df2[df2[present_cols].isnull().any(axis=1)].copy() if present_cols else df2.iloc[0:0].copy()
        df2_kept = df2.dropna(subset=present_cols) if present_cols else df2.copy()
        after = df2_kept.shape[0]

        print(f"Dropped {before - after} rows with NaNs in required columns (remaining {after} rows).")
        
        if not dropped_rows.empty:
             try:
                dropped_rows.to_csv('dropped_rows_inspection.csv', index=False)
                print("Saved dropped rows to 'dropped_rows_inspection.csv'.")
             except Exception as e:
                print("Failed to save dropped_rows_inspection.csv:", e)

        return df2_kept

    def remove_outliers(df, cols, target_col):
        """
        Removes rows containing outliers in the specified columns using the Z-score method.
        """
        df_cleaned = df.copy()
        cols_to_check = cols + [target_col]
        present_cols = [c for c in cols_to_check if c in df_cleaned.columns]
        
        # Ensure we don't try to z-score constant or empty columns
        for col in present_cols:
            # coerce object columns to numeric where possible (should be handled by previous step but good safety)
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

    # --- Apply Preprocessing ---
    print("--- Data Preprocessing ---")
    data_numeric = coerce_and_clean_numeric(data, feature_cols, target_col, group_col)
    data_cleaned = remove_outliers(data_numeric, feature_cols, target_col)
    print(f"Original shape: {data.shape}")
    print(f"Final data shape: {data_cleaned.shape}")

    if data_cleaned.empty:
        print("Error: No data remaining after cleaning.")
        return

    if group_col not in data_cleaned.columns:
        print(f"Error: Group column '{group_col}' missing.")
        return

    # --- Prepare for MERF ---
    
    # X: Fixed effects features
    X = data_cleaned[feature_cols]
    
    # Z: Random effects features (Random Intercept -> ones)
    Z = np.ones((len(data_cleaned), 1))
    
    # clusters: Grouping variable
    clusters = data_cleaned[group_col]
    
    # y: Target variable
    y = data_cleaned[target_col]

    # Split data
    # Note: For strict evaluation, splitting should ideally respect clusters (e.g. GroupShuffleSplit),
    # but for consistency with previous scripts (Random_Forest_Code.py), we'll use train_test_split.
    # MERF handles new clusters in test set by using fixed effects only (random effect = 0).
    
    X_train, X_test, y_train, y_test, Z_train, Z_test, clusters_train, clusters_test = train_test_split(
        X, y, Z, clusters, test_size=0.2, random_state=42
    )

    print("\n--- Fitting Mixed Effect Random Forest ---")
    
    # Initialize MERF
    # n_estimators=100 consistent with Random_Forest_Code.py
    # max_iterations=100 for consistency with 100 epochs
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    merf_model = MERF(fixed_effects_model=rf, max_iterations=100)
    
    # Fit the model
    # merf.fit(X, Z, clusters, y)
    merf_model.fit(X_train, Z_train, clusters_train, y_train)

    print("\n--- Model Evaluation ---")
    
    # Predict
    # merf.predict(X, Z, clusters)
    y_pred = merf_model.predict(X_test, Z_test, clusters_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"MERF Mean Squared Error: {mse}")
    print(f"MERF Root Mean Squared Error: {rmse}")

    # Plotting Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Chlorophyll Depth')
    plt.ylabel('Predicted Chlorophyll Depth')
    plt.title('Mixed Effect Random Forest: Actual vs Predicted Chlorophyll Depth')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_filename = 'mixed_random_forest_prediction_scatter.png'
    plt.savefig(plot_filename)
    print(f"Scatter plot saved to '{plot_filename}'")
    plt.show()

    # Training convergence plot (optional, MERF usually provides loss history)
    if hasattr(merf_model, 'gll_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(merf_model.gll_history)
        plt.xlabel('Iteration')
        plt.ylabel('Generalized Log Likelihood')
        plt.title('MERF Training Convergence')
        plt.grid(True)
        # plt.show() # Not showing, just noting it's possible

if __name__ == "__main__":
    main()
