import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.data.loader import DataLoader
from src.utils.logger import setup_logger
from src.utils.config import MODEL_PATH, METADATA_PATH

logger = setup_logger("model_evaluation")

def run_evaluation():
    logger.info("Running custom model evaluation diagnostic...")
    
    if not MODEL_PATH.exists():
        logger.error(f"Model path {MODEL_PATH} does not exist. Please train the model first.")
        return
        
    # Load pipeline
    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline["preprocessor"]
    rf_model = pipeline["model"]
    
    # Load applications
    loader = DataLoader()
    df = loader.load_applications()
    
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    
    # Transform
    X_proc = preprocessor.transform(X)
    
    # Predict
    probs = rf_model.predict_proba(X_proc)[:, 1]
    preds = rf_model.predict(X_proc)
    
    # Report metrics
    auc_score = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, target_names=["Repaid (0)", "Default (1)"])
    
    print("\n" + "="*50)
    print("      AI CREDIT RISK PLATFORM MODEL DIAGNOSTIC")
    print("="*50)
    print(f"Total Applications Scored: {len(df)}")
    print(f"Model ROC-AUC Score:      {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_evaluation()
