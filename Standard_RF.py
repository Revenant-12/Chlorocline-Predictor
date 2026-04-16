#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import tkinter as tk
from tkinter import messagebox, filedialog
import argparse
import sys
import os

# Load data
try:
    data = pd.read_csv('Raw data chlorocline.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'Raw data chlorocline.csv' not found.")
    exit()

# Standardized Feature and target columns
feature_cols = ['shallow_depth_s', 'deep_depth_s', 'Deep_depth_t']
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

def create_gui(rf_model, feature_cols):
    root = tk.Tk()
    root.title("Standardized RF Chlorocline Predictor")
    
    # Configure grid weights for resizability
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    
    # Define fonts
    main_font = ("Helvetica", 14)
    bold_font = ("Helvetica", 14, "bold")
    result_font = ("Helvetica", 18, "bold")

    # Standardized Labels and Entries
    labels = ["Start of the Halocline:", "End of the Halocline:", "Temp Depth at Deep End:"]
    entries = []

    for i, text in enumerate(labels):
        root.rowconfigure(i, weight=1)
        tk.Label(root, text=text, font=main_font).grid(row=i, column=0, padx=20, pady=10, sticky="e")
        entry = tk.Entry(root, font=main_font)
        entry.grid(row=i, column=1, padx=20, pady=10, sticky="ew")
        entries.append(entry)

    def predict():
        try:
            # Inputs: shallow_depth_s, deep_depth_s, Deep_depth_t
            inputs = [float(e.get()) for e in entries]
            # Convert to 2D array for scikit-learn
            prediction = rf_model.predict([inputs])[0]
            
            result_display.config(state="normal")
            result_display.delete("1.0", tk.END)
            result_display.insert(tk.END, f"Predicted Chlorophyll Depth: {prediction:.2f}")
            result_display.config(state="disabled")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def reset():
        for e in entries:
            e.delete(0, tk.END)
        result_display.config(state="normal")
        result_display.delete("1.0", tk.END)
        result_display.insert(tk.END, "Predicted Chlorophyll Depth: -")
        result_display.config(state="disabled")

    # Buttons
    btn_row = len(labels)
    root.rowconfigure(btn_row, weight=1)
    predict_btn = tk.Button(root, text="Predict", command=predict, font=bold_font)
    predict_btn.grid(row=btn_row, column=0, pady=20, padx=20, sticky="ew")

    reset_btn = tk.Button(root, text="Reset", command=reset, font=bold_font)
    reset_btn.grid(row=btn_row, column=1, pady=20, padx=20, sticky="ew")

    # Result Display
    res_row = btn_row + 1
    root.rowconfigure(res_row, weight=1)
    result_display = tk.Text(root, font=result_font, height=1, bg=root.cget("bg"), borderwidth=0, highlightthickness=0)
    result_display.tag_configure("center", justify='center')
    result_display.insert(tk.END, "Predicted Chlorophyll Depth: -")
    result_display.tag_add("center", "1.0", "end")
    result_display.config(state="disabled")
    result_display.grid(row=res_row, column=0, columnspan=2, pady=20, sticky="nsew")

    # Batch CSV Section
    batch_row = res_row + 1
    root.rowconfigure(batch_row, weight=1)
    
    def load_csv():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            required = ['shallow_depth_s', 'deep_depth_s', 'Deep_depth_t']
            if not all(col in df.columns for col in required):
                messagebox.showerror("Column Error", f"CSV must contain {required} columns.")
                return
            
            # Fill entries with first row data
            for i, col in enumerate(required):
                entries[i].delete(0, tk.END)
                entries[i].insert(0, str(df[col].iloc[0]))
            
            # Perform predictions for all rows
            X_batch = df[required].values
            df['Prediction'] = rf_model.predict(X_batch)
            
            # Save updated CSV
            df.to_csv(file_path, index=False)
            predict()
            messagebox.showinfo("Success", f"Predictions written to 'Prediction' column in:\n{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process CSV: {e}")

    batch_btn = tk.Button(root, text="Load CSV for Batch Prediction", command=load_csv, font=bold_font, bg="lightgrey")
    batch_btn.grid(row=batch_row, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

    print("Starting interactive GUI...")
    root.mainloop()

def run_cli(rf_model, args):
    try:
        inputs = [args.shallow, args.deep_s, args.deep_t]
        prediction = rf_model.predict([inputs])[0]
        print(f"\n--- Prediction ---")
        print(f"Shallow Depth (S): {args.shallow}")
        print(f"Deep Depth (S): {args.deep_s}")
        print(f"Deep Depth (T): {args.deep_t}")
        print(f"Predicted Chlorophyll Depth: {prediction:.2f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Start the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardized Random Forest Predictor")
    parser.add_argument("--shallow", type=float, help="shallow_depth_s")
    parser.add_argument("--deep_s", type=float, help="deep_depth_s")
    parser.add_argument("--deep_t", type=float, help="Deep_depth_t")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode")

    args = parser.parse_args()

    # Apply cleaning and outlier removal
    print("--- Data Preprocessing ---")
    data_numeric = coerce_and_clean_numeric(data, feature_cols, target_col)
    data_cleaned = remove_outliers(data_numeric)

    # Prepare Feature and Target sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_cleaned[feature_cols].values, 
        data_cleaned[target_col].values, 
        test_size=0.2, 
        random_state=42
    )

    print("\n--- Model Training ---")
    # Standardized parameters: 100 estimators (epochs equivalent)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("\n--- Model Evaluation ---")
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Standard RF Mean Squared Error: {mse}")
    print(f"Standard RF Root Mean Squared Error: {rmse}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Chlorophyll Depth')
    plt.ylabel('Predicted Chlorophyll Depth')
    plt.title('Standard Random Forest: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.savefig('standard_rf_prediction_scatter.png')
    plt.close()
    print("Evaluation plot saved to 'standard_rf_prediction_scatter.png'")

    # If any numeric argument is provided, run in CLI mode
    if args.shallow is not None and args.deep_s is not None and args.deep_t is not None:
        run_cli(rf_model, args)
    elif args.cli:
        parser.print_help()
    else:
        create_gui(rf_model, feature_cols)
