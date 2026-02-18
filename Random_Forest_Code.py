#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import tkinter as tk
from tkinter import messagebox
import argparse
import sys

# Load data
try:
    data = pd.read_csv('Raw data chlorocline.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'Raw data chlorocline.csv' not found.")
    exit()

# Feature and target columns
feature_cols = ['shallow_depth_s', 'Deep_depth_s']
target_col = 'chlorophyll_depth'

def remove_outliers(df):
    """
    Removes rows containing outliers in the specified columns using the Z-score method.
    """
    df_cleaned = df.copy()
    cols_to_check = feature_cols + [target_col]
    present_cols = [c for c in cols_to_check if c in df_cleaned.columns]
    
    # Ensure we don't try to z-score constant or empty columns
    for col in present_cols:
        # coerce object columns to numeric where possible
        if df_cleaned[col].dtype == object:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        if df_cleaned[col].dropna().nunique() <= 1:
            continue
        # compute zscore safely
        try:
            z = stats.zscore(df_cleaned[col].astype(float))
        except Exception:
            continue
        outlier_mask = np.abs(z) > 2  # adjust threshold if desired
        df_cleaned = df_cleaned[~outlier_mask]

    return df_cleaned


def coerce_and_clean_numeric(df, cols, target_col):
    """Coerce listed columns to numeric, replace infs, drop rows with NaNs, and provide diagnostics.

    Returns the cleaned DataFrame (rows kept). Also saves dropped rows to 'dropped_rows_inspection.csv'.
    """
    df2 = df.copy()
    converted = {}
    coercion_examples = {}

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

    present_cols = [c for c in cols + [target_col] if c in df2.columns]
    before = df2.shape[0]

    # identify rows that will be dropped and keep a copy for diagnostics
    dropped_rows = df2[df2[present_cols].isnull().any(axis=1)].copy() if present_cols else df2.iloc[0:0].copy()
    df2_kept = df2.dropna(subset=present_cols) if present_cols else df2.copy()
    after = df2_kept.shape[0]

    print(f"Dropped {before - after} rows with NaNs in required columns (remaining {after} rows).")

    if not dropped_rows.empty:
        # summarize drop reasons per-row
        def reasons(row):
            return [col for col in present_cols if pd.isna(row[col])]
        dropped_rows['drop_reasons'] = dropped_rows.apply(reasons, axis=1)
        # save dropped rows for inspection
        try:
            dropped_rows.to_csv('dropped_rows_inspection.csv', index=False)
            print("Saved dropped rows to 'dropped_rows_inspection.csv' (first 5 rows):")
        except Exception as e:
            print("Failed to save dropped_rows_inspection.csv:", e)
        display_cols = [c for c in present_cols if c in dropped_rows.columns]
        print(dropped_rows.head()[display_cols + ['drop_reasons']])

    return df2_kept

# Apply cleaning and outlier removal
print("--- Data Preprocessing ---")
data_numeric = coerce_and_clean_numeric(data, feature_cols, target_col)
data_cleaned = remove_outliers(data_numeric)

print("Original DataFrame shape:", data.shape)
print("DataFrame shape after coercion+outlier removal:", data_cleaned.shape)

# Prepare Feature and Target sets
X = data_cleaned[feature_cols].values
y = data_cleaned[target_col].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Model Training ---")
# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=86,
    max_depth=5,
    max_features=3, 
    random_state=42
    )

# Train the model
rf_model.fit(X_train, y_train)

print("\n--- Model Evaluation ---")
# Predict on test set
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Random Forest Mean Squared Error: {mse}")
print(f"Random Forest Root Mean Squared Error: {rmse}")

# Feature Importance
print("\n--- Feature Importance ---")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(len(feature_cols)):
    print(f"{feature_cols[indices[i]]}: {importances[indices[i]]}")

# Plotting Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Chlorophyll Depth')
plt.ylabel('Predicted Chlorophyll Depth')
plt.title('Random Forest: Actual vs Predicted Chlorophyll Depth')
plt.legend()
plt.grid(True)
plt.savefig('rf_prediction_scatter.png')
plt.close()
print("Evaluation plot saved to 'rf_prediction_scatter.png'")

# --- Model Export and Interactive UI ---

# Export the trained model
joblib.dump(rf_model, 'rf_model.joblib')
print("\nModel exported to 'rf_model.joblib'")

def create_gui():
    root = tk.Tk()
    root.title("Chlorocline Predictor")

    # Labels and Entries
    labels = ["Start of the Halocline:", "End of the Halocline:"]
    entries = []

    for i, text in enumerate(labels):
        tk.Label(root, text=text).grid(row=i, column=0, padx=10, pady=5, sticky="e")
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries.append(entry)

    def predict():
        try:
            # Inputs: Start of Halocline, End of Halocline, End of Thermocline
            inputs = [float(e.get()) for e in entries]
            # Convert to 2D array for scikit-learn
            prediction = rf_model.predict([inputs])[0]
            result_label.config(text=f"Predicted Chlorophyll Depth: {prediction:.2f}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def reset():
        for e in entries:
            e.delete(0, tk.END)
        result_label.config(text="Predicted Chlorophyll Depth: -")

    # Buttons
    predict_btn = tk.Button(root, text="Predict", command=predict)
    predict_btn.grid(row=len(labels), column=0, pady=10)

    reset_btn = tk.Button(root, text="Reset", command=reset)
    reset_btn.grid(row=len(labels), column=1, pady=10)

    # Result Label
    result_label = tk.Label(root, text="Predicted Chlorophyll Depth: -", font=("Helvetica", 10, "bold"))
    result_label.grid(row=len(labels)+1, column=0, columnspan=2, pady=10)

    print("Starting interactive GUI...")
    root.mainloop()

def run_cli(args):
    try:
        inputs = [args.start_halo, args.end_halo, args.end_thermo]
        prediction = rf_model.predict([inputs])[0]
        print(f"\n--- Prediction ---")
        print(f"Start of the Halocline: {args.start_halo}")
        print(f"End of the Halocline: {args.end_halo}")
        print(f"End of the Thermocline: {args.end_thermo}")
        print(f"Predicted Chlorophyll Depth: {prediction:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Start the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chlorocline Predictor - Random Forest")
    parser.add_argument("--start_halo", type=float, help="Start of the Halocline")
    parser.add_argument("--end_halo", type=float, help="End of the Halocline")
    parser.add_argument("--end_thermo", type=float, help="End of the Thermocline")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode (requires inputs)")

    args = parser.parse_args()

    # If any numeric argument is provided, run in CLI mode
    if args.start_halo is not None and args.end_halo is not None and args.end_thermo is not None:
        run_cli(args)
    elif args.cli:
        print("Error: CLI mode requires --start_halo, --end_halo, and --end_thermo")
        parser.print_help()
    else:
        # Default to GUI
        create_gui()

    