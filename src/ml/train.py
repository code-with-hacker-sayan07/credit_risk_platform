import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from src.data.loader import DataLoader
from src.data.preprocessor import CreditPreprocessor
from src.utils.logger import setup_logger
from src.utils.config import MODEL_PATH, METADATA_PATH

logger = setup_logger("model_training")

def train_credit_model():
    logger.info("Initializing data loader and database...")
    loader = DataLoader()
    loader.initialize_db()
    
    logger.info("Loading applications data...")
    df = loader.load_applications()
    
    if df.empty:
        logger.error("No application data found in database. Exiting...")
        return
        
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    # Train-test split
    logger.info("Splitting dataset into train and test sets (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Fit preprocessor
    logger.info("Fitting feature engineering and preprocessing pipeline...")
    preprocessor = CreditPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # Train Random Forest with class balancing
    logger.info("Training class-balanced Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_proc, y_train)
    
    # Evaluate performance
    logger.info("Evaluating trained model performance...")
    test_probs = rf_model.predict_proba(X_test_proc)[:, 1]
    roc_auc = roc_auc_score(y_test, test_probs)
    
    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    pr_auc = auc(recall, precision)
    
    test_preds = (test_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    
    logger.info(f"Model Evaluation Metrics:")
    logger.info(f" - ROC-AUC Score: {roc_auc:.4f}")
    logger.info(f" - PR-AUC Score:  {pr_auc:.4f}")
    logger.info(f" - Confusion Matrix:\n{cm}")
    
    # Extract Feature Importances
    importances = rf_model.feature_importances_
    features = preprocessor.get_feature_names()
    
    feature_importances = sorted(
        [{"feature": f, "importance": float(imp)} for f, imp in zip(features, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    # Bundle preprocessor and model together for seamless serialization
    model_pipeline = {
        "preprocessor": preprocessor,
        "model": rf_model,
        "features": features
    }
    
    logger.info(f"Saving compiled model pipeline to {MODEL_PATH}...")
    joblib.dump(model_pipeline, MODEL_PATH)
    
    # Save performance metadata for UI reference
    # Convert arrays to list for JSON serialization
    pr_curve = [{"recall": float(r), "precision": float(p)} for r, p in zip(recall[::5], precision[::5])] # downsample for size
    
    # Generate mock ROC curve data points
    fpr, tpr, _ = roc_curve_data(y_test, test_probs)
    roc_curve = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr[::5], tpr[::5])]
    
    metadata = {
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist(),
            "default_rate": float(y.mean()),
            "total_records": len(df)
        },
        "feature_importances": feature_importances[:20], # top 20
        "curves": {
            "roc": roc_curve,
            "pr": pr_curve
        }
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Feature metadata and metrics saved successfully to {METADATA_PATH}.")

def roc_curve_data(y_true, y_score):
    """Simple calculation of False Positive Rate and True Positive Rate curves."""
    thresholds = np.sort(y_score)
    fprs = []
    tprs = []
    
    P = sum(y_true == 1)
    N = sum(y_true == 0)
    
    for t in thresholds:
        tp = sum((y_score >= t) & (y_true == 1))
        fp = sum((y_score >= t) & (y_true == 0))
        
        tprs.append(tp / P if P > 0 else 0)
        fprs.append(fp / N if N > 0 else 0)
        
    # Append final endpoints
    fprs.append(0.0)
    tprs.append(0.0)
    fprs.reverse()
    tprs.reverse()
    
    # Ensure they start at 0 and end at 1
    return [0.0] + fprs + [1.0], [0.0] + tprs + [1.0], None

if __name__ == "__main__":
    train_credit_model()
