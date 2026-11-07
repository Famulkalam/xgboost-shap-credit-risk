"""Exploratory data analysis on the cleaned dataset.

Run:  python eda.py   (writes plots to figures/)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_prep import TARGET, clean, load_raw

FIG_DIR = "figures"


def run_eda():
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw()

    print("--- Raw data ---")
    print(f"Shape: {raw.shape}")
    print("Missing values (%):")
    print((raw.isna().mean() * 100).round(2)[raw.isna().any()])
    print("Ranges that need cleaning:")
    print(raw[["person_age", "person_emp_length", "person_income"]].describe().loc[["min", "max"]])

    df, log = clean(raw)
    print("\n--- Cleaning log ---")
    for k, v in log.items():
        print(f"{k}: {v}")

    print("\n--- Default rate by loan grade ---")
    print(df.groupby("loan_grade")[TARGET].agg(["mean", "size"]).round(3))

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=TARGET, data=df, color="#4C72B0")
    ax.set_xticks([0, 1], ["Repaid (0)", "Defaulted (1)"])
    plt.title(f"Default rate {df[TARGET].mean():.1%} (n = {len(df):,})")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/default_distribution.png", dpi=150)
    plt.close()

    features = ["person_income", "loan_percent_income", "loan_int_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in zip(axes, features):
        sns.boxplot(x=TARGET, y=col, data=df, ax=ax, color="#DD8452", showfliers=False)
        ax.set_title(f"{col} by loan status")
        if col == "person_income":
            ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/key_features_boxplots.png", dpi=150)
    plt.close()

    corr = df.select_dtypes(include=[np.number]).drop(columns="loan_id").corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation of numeric variables (cleaned data)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {FIG_DIR}/")


if __name__ == "__main__":
    run_eda()
