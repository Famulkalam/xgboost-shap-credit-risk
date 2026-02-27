import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config
DATA_PATH = "credit_risk_dataset.csv"  # Place your Kaggle CSV here

def run_eda(path):
    print("--- 1. Loading Dataset ---")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} not found. Please ensure the Kaggle dataset is in the same directory.")
        return

    # --- 2. Basic Info ---
    print("\n--- 2. Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values (%):")
    print(df.isnull().sum() / len(df) * 100)

    # --- 3. Summary Statistics ---
    print("\n--- 3. Numerical Summary Statistics ---")
    print(df.describe())

    # --- 4. Default Analysis ---
    print("\n--- 4. Target Variable (loan_status) Distribution ---")
    default_counts = df['loan_status'].value_counts(normalize=True) * 100
    print(f"Non-Default (0): {default_counts[0]:.2f}%")
    print(f"Default (1): {default_counts[1]:.2f}%")

    # --- 5. Correlations ---
    print("\n--- 5. Correlation Analysis ---")
    # Mapping categorical if needed for correlation (simple numeric approach)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    print("Top Correlations with loan_status:")
    print(corr_matrix['loan_status'].sort_values(ascending=False))

    # --- 6. Plotting ---
    print("\n--- 6. Generating Plots ---")
    sns.set_theme(style="whitegrid")

    # A. Histogram of Loan Status
    plt.figure(figsize=(6, 4))
    sns.countplot(x='loan_status', data=df, palette='viridis')
    plt.title('Distribution of Loan Defaults (0=Good, 1=Bad)')
    plt.savefig('default_distribution.png')
    plt.close()

    # B. Box Plots of Key Features split by Default
    features_to_plot = ['person_age', 'person_income', 'loan_percent_income']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(features_to_plot):
        sns.boxplot(x='loan_status', y=col, data=df, ax=axes[i], palette='magma')
        axes[i].set_title(f'{col} vs Loan Status')
    plt.tight_layout()
    plt.savefig('key_features_boxplots.png')
    plt.close()

    # C. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    print("EDA Complete! Plots saved as 'default_distribution.png', 'key_features_boxplots.png', and 'correlation_heatmap.png'.")

if __name__ == "__main__":
    run_eda(DATA_PATH)
