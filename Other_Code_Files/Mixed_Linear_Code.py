#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
    feature_cols = ['shallow_depth_s', 'Deep_depth_s', 'Deep_depth_t']
    target_col = 'chlorophyll_depth'
    # We use 'Site' for the mixed model grouping variable
    group_col = 'Site'

    # --- Preprocessing Functions from Notebook ---

    def coerce_and_clean_numeric(df, cols, target_col):
        """Coerce listed columns to numeric, replace infs, drop rows with NaNs, and provide diagnostics. 
        
        Returns the cleaned DataFrame (rows kept). Also saves dropped rows to 'dropped_rows_inspection.csv'.
        """
        df2 = df.copy()
        converted = {}
        coercion_examples = {}

        # Only coerce the numeric feature/target columns, leave 'Site' alone
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

        # We need to make sure we don't drop rows just because of other columns, 
        # but we DO need Site to be present for the mixed model.
        cols_to_check = cols + [target_col]
        if group_col in df2.columns:
            cols_to_check.append(group_col)
        
        present_cols = [c for c in cols_to_check if c in df2.columns]
        
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
    data_numeric = coerce_and_clean_numeric(data, feature_cols, target_col)
    data_cleaned = remove_outliers(data_numeric)
    print(f"Original shape: {data.shape}")
    print(f"Final data shape: {data_cleaned.shape}")

    # --- Fit Mixed Linear Model ---
    print("\n--- Fitting Mixed Linear Model ---")

    # Formula: target ~ features
    # Note: MixedLM expects formula typically.
    formula = f"{target_col} ~ {' + '.join(feature_cols)}"

    print(f"Formula: {formula}")
    print(f"Group variable: {group_col}")

    if group_col not in data_cleaned.columns:
        print(f"Error: Grouping column '{group_col}' not found in data.")
        return

    try:
        # Check if we have enough groups
        n_groups = data_cleaned[group_col].nunique()
        print(f"Number of groups ({group_col}): {n_groups}")
        
        if n_groups < 1:
             print("Error: No groups found.")
             return

        # Fit the model
        # Using simple random intercept for groups
        model = smf.mixedlm(formula, data_cleaned, groups=data_cleaned[group_col])
        result = model.fit()
        
        print("Model Converged:", result.converged)
        print(result.summary())
        
        # Save summary to file
        output_file = "mixed_model_summary.txt"
        with open(output_file, "w") as f:
            f.write(result.summary().as_text())
        print(f"\nModel summary saved to '{output_file}'")

        # Plotting Fitted vs Actual (Training data)
        y_actual = data_cleaned[target_col]
        y_fitted = result.fittedvalues
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_fitted, color='blue', alpha=0.5, label='Fitted vs Actual')
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=2, label='Perfect Fit')
        plt.xlabel('Actual Chlorophyll Depth')
        plt.ylabel('Fitted Chlorophyll Depth')
        plt.title('Mixed Linear Model: Actual vs Fitted Chlorophyll Depth')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"\nError fitting Mixed Linear Model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()