import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Config
DATA_PATH = "credit_risk_dataset.csv"
PREDICTIONS_CSV = "predictions.csv"
PROB_THRESHOLD = 0.7  # Define what counts as 'high risk'

def train_credit_model():
    # --- 1. Load and Clean ---
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # HistGradientBoosting handles NaNs natively, so minimal filling needed 
    # but we will fill for consistent EDA/SHAP experience
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # --- 2. Feature Engineering ---
    print("Engineering features...")
    # A. Debt-to-Income (approximation if we had total debt, here using loan/income)
    df['loan_to_income'] = df['loan_amnt'] / df['person_income']
    
    # B. Age Bins
    df['age_group'] = pd.cut(df['person_age'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Senior', 'Elderly'])
    
    # C. Intent and Income Interaction
    # (Just an example of interaction)
    
    # --- 3. Encoding Categorical Data ---
    print("Encoding categorical variables...")
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file', 'age_group']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # --- 4. Split Data ---
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # --- 5. Train HistGradientBoosting ---
    print("Training HistGradientBoosting Model (XGBoost/LightGBM style)...")
    model = HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.05,
        max_depth=6,
        scoring='roc_auc',
        early_stopping=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # --- 6. Predictions & Evaluation ---
    # Test Metrics
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)

    # Train Metrics (For Overfitting Check)
    y_train_pred_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_pred_prob > 0.5).astype(int)

    print("\n--- Model Evaluation (Overfitting Check) ---")
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_pred_prob)
    test_auc = roc_auc_score(y_test, y_test_pred_prob)

    print(f"Train Accuracy: {train_acc:.4f}  |  Test Accuracy: {test_acc:.4f}")
    print(f"Train AUC-ROC:  {train_auc:.4f}  |  Test AUC-ROC:  {test_auc:.4f}")
    
    overfit_gap = train_auc - test_auc
    if overfit_gap > 0.05:
        print(f"\nWARNING: Potential Overfitting (Gap: {overfit_gap:.4f}). Model performs significantly better on training data.")
    else:
        print(f"\nModel is ROBUST (Gap: {overfit_gap:.4f}). Performance is consistent across Train and Test sets.")

    print("\n--- Detailed Test Metrics ---")
    print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # --- 7. SHAP Explanations ---
    print("\nGenerating SHAP explanations...")
    # For HistGradientBoosting, we use Explainer with a masker
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test, check_additivity=False)

    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('shap_summary_plot.png')
    plt.close()

    # Waterfall Plot for top 3 highest-risk instances
    top_risk_indices = np.argsort(y_test_pred_prob)[-3:]
    for i, idx in enumerate(top_risk_indices):
        plt.figure(figsize=(10, 6))
        # Ensure we use the correct shap values for the observation
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.savefig(f'shap_waterfall_loan_{idx}.png')
        plt.close()

    # --- 8. Save Predictions with Risk Bands ---
    print(f"Saving predictions to {PREDICTIONS_CSV}...")
    results_df = X_test.copy()
    results_df['probability_default'] = y_test_pred_prob
    results_df['is_high_risk'] = (y_test_pred_prob > PROB_THRESHOLD).astype(int)
    
    # Assign Risk Bands
    def assign_risk_band(prob):
        if prob < 0.3: return 'Low'
        if prob < 0.7: return 'Medium'
        return 'High'
    
    results_df['risk_band'] = results_df['probability_default'].apply(assign_risk_band)
    
    # Assuming we have an original row index or ID to map back to SQL
    results_df.to_csv(PREDICTIONS_CSV, index=True) 

    print("Success! Model trained, SHAP plots saved, and predictions exported.")

if __name__ == "__main__":
    train_credit_model()
