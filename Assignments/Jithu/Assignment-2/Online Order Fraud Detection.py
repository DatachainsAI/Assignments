import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')
RND = 42
np.random.seed(RND)

# --- Config ---
input_csv = 'data/ecommerce_fraud_data.csv' 
output_dir = './output'                           
train_models_flag = True                          
threshold = 0.5 


# --------------------------------------------------------
#  FEATURE ENGINEERING
# --------------------------------------------------------
def feature_engineering(df):
    df = df.copy()

    # Convert order date
    if 'order_date' in df.columns and df['order_date'].dtype == object:
        try:
            df['order_date'] = pd.to_datetime(df['order_date'])
        except Exception:
            pass

    # user-level aggregates
    if 'user_id' in df.columns:
        user_stats = df.groupby('user_id').agg(
            user_mean_order=('order_amount', 'mean'),
            user_order_count=('user_id', 'count'),
            user_device_changes=('device_change_count', 'max')
        ).reset_index()
        df = df.merge(user_stats, on='user_id', how='left')

    # derived features
    if 'user_mean_order' in df.columns:
        df['order_value_deviation'] = (
            (df['order_amount'] - df['user_mean_order']).abs() / (df['user_mean_order'] + 1e-6)
        )

    if 'order_date' in df.columns:
        df['order_hour'] = df['order_date'].dt.hour
        df['order_weekday'] = df['order_date'].dt.weekday

    df['high_device_change'] = (df.get('device_change_count', 0) > 2).astype(int)

    # recent order velocity
    df['high_velocity'] = (
        (
            (df['recent_orders_1day'] >= 3) if 'recent_orders_1day' in df else pd.Series([0] * len(df))
        ) |
        (
            (df['recent_orders_1hr'] >= 2) if 'recent_orders_1hr' in df else pd.Series([0] * len(df))
        )
    ).astype(int)

    # Categorical encoding
    cat_cols = [c for c in ['billing_region', 'shipping_region', 'device_type', 'payment_type'] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


# --------------------------------------------------------
#  FEATURE PREPARATION
# --------------------------------------------------------
def prepare_features(df):
    exclude_cols = ['order_id', 'user_id', 'product_id', 'order_date', 'fraud_prob_sim', 'is_fraud']
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    num_cols = df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols].fillna(0)
    return X


# --------------------------------------------------------
#  TRAIN MODELS
# --------------------------------------------------------
def train_models(X_train, y_train):
    xgb = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss', random_state=RND
    )
    xgb.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=RND)
    iso.fit(X_train_scaled)

    return xgb, iso, scaler


# --------------------------------------------------------
#  EVALUATION, PLOTTING & MODEL SAVING
# --------------------------------------------------------
def evaluate_and_save(xgb, iso, scaler, X_test, y_test, df_test, output_dir, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    # --- XGBoost predictions ---
    y_proba = xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # --- Isolation Forest scores ---
    X_test_scaled = scaler.transform(X_test)
    iso_scores = iso.decision_function(X_test_scaled)
    iso_anomaly = -iso_scores
    iso_norm = (iso_anomaly - iso_anomaly.min()) / (iso_anomaly.max() - iso_anomaly.min() + 1e-9)
    iso_flag = (iso_norm > 0.5).astype(int)

    # --- Save models ---
    joblib.dump(xgb, os.path.join(output_dir, 'xgb_model.joblib'))
    joblib.dump(iso, os.path.join(output_dir, 'iso_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print(f"\n💾 Models saved to {output_dir}")

    # --- Metrics ---
    auc_xgb = roc_auc_score(y_test, y_proba)
    auc_iso = roc_auc_score(y_test, iso_norm)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    print("\n📈 Model Performance Metrics")
    print(f"XGBoost AUC: {auc_xgb:.3f}")
    print(f"Isolation Forest AUC: {auc_iso:.3f}")
    print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    # --- Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_xgb = confusion_matrix(y_test, y_pred)
    cm_iso = confusion_matrix(y_test, iso_flag)

    ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Not Fraud', 'Fraud']).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("XGBoost Confusion Matrix")

    ConfusionMatrixDisplay(confusion_matrix=cm_iso, display_labels=['Not Fraud', 'Fraud']).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Isolation Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.show()

    # --- ROC Curves ---
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba)
    fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_norm)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.3f})')
    plt.plot(fpr_iso, tpr_iso, label=f'Isolation Forest (AUC={auc_iso:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.show()

    # --- Classification Reports ---
    print("\n📊 XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 Isolation Forest Classification Report:")
    print(classification_report(y_test, iso_flag))

    # --- Fraud Risk Heatmaps ---
    df_test['xgb_flag'] = y_pred
    df_test['iso_flag'] = iso_flag

    # Region-wise Heatmap (XGBoost)
    if 'billing_region' in df_test.columns:
        plt.figure(figsize=(8, 5))
        region_risk = df_test.groupby('billing_region')['xgb_flag'].mean().sort_values(ascending=False)
        sns.heatmap(region_risk.to_frame().T, annot=True, cmap='Reds')
        plt.title('Region-wise Fraud Risk (XGBoost)')
        plt.savefig(os.path.join(output_dir, 'region_fraud_heatmap_xgb.png'))
        plt.show()

    # Device-wise Heatmap (XGBoost)
    if 'device_type' in df_test.columns:
        plt.figure(figsize=(8, 5))
        device_risk = df_test.groupby('device_type')['xgb_flag'].mean().sort_values(ascending=False)
        sns.heatmap(device_risk.to_frame().T, annot=True, cmap='Reds')
        plt.title('Device-wise Fraud Risk (XGBoost)')
        plt.savefig(os.path.join(output_dir, 'device_fraud_heatmap_xgb.png'))
        plt.show()


# --------------------------------------------------------
#  MAIN PIPELINE
# --------------------------------------------------------
def run_pipeline():
    print(f"Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    df['is_fraud'] = df['is_fraud'].map({'No': 0, 'Yes': 1})
    df_fe = feature_engineering(df)
    X = prepare_features(df_fe)

    if train_models_flag:
        y = df['is_fraud']
        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X, y, df, test_size=0.2, stratify=y, random_state=RND
        )

        xgb, iso, scaler = train_models(X_train, y_train)
        evaluate_and_save(xgb, iso, scaler, X_test, y_test, df_test, output_dir, threshold)
        print("✅ Training complete. All artifacts and plots saved to:", output_dir)

    else:
        xgb = joblib.load(os.path.join(output_dir, 'xgb_model.joblib'))
        iso = joblib.load(os.path.join(output_dir, 'iso_model.joblib'))
        scaler = joblib.load(os.path.join(output_dir, 'scaler.joblib'))

        X_scaled = scaler.transform(X)
        proba = xgb.predict_proba(X)[:, 1]
        iso_scores = -iso.decision_function(X_scaled)
        iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)

        out = X.copy()
        out['xgb_proba'] = proba
        out['xgb_flag'] = (proba >= threshold).astype(int)
        out['iso_score'] = iso_norm
        out.to_csv(os.path.join(output_dir, 'scored_orders.csv'), index=False)
        print("✅ Predictions saved to:", os.path.join(output_dir, 'scored_orders.csv'))


if __name__ == '__main__':
    run_pipeline()
