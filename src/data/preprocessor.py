import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to perform credit-specific feature engineering."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Calculate derived ratio metrics
        X_out['CREDIT_TO_INCOME_RATIO'] = X_out['AMT_CREDIT'] / (X_out['AMT_INCOME_TOTAL'] + 1e-5)
        X_out['ANNUITY_TO_INCOME_RATIO'] = X_out['AMT_ANNUITY'] / (X_out['AMT_INCOME_TOTAL'] + 1e-5)
        X_out['GOODS_PRICE_TO_CREDIT_RATIO'] = X_out['AMT_GOODS_PRICE'] / (X_out['AMT_CREDIT'] + 1e-5)
        
        # Convert Days demographics to absolute positive years
        X_out['AGE_YEARS'] = -X_out['DAYS_BIRTHDAY'] / 365.25
        
        # Handle Days Employed (Pensioner has 365243)
        X_out['EMPLOYED_YEARS'] = X_out['DAYS_EMPLOYED'].apply(
            lambda x: 0.0 if x >= 365243 or x is None else -x / 365.25
        )
        X_out['EMPLOYMENT_TO_AGE_RATIO'] = X_out['EMPLOYED_YEARS'] / (X_out['AGE_YEARS'] + 1e-5)
        
        # Drop raw high-cardinality or raw dates columns that have been processed
        cols_to_drop = ['DAYS_BIRTHDAY', 'DAYS_EMPLOYED']
        for col in cols_to_drop:
            if col in X_out.columns:
                X_out = X_out.drop(columns=[col])
                
        return X_out

class CreditPreprocessor:
    def __init__(self):
        self.numerical_cols = [
            'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
            'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO',
            'GOODS_PRICE_TO_CREDIT_RATIO', 'AGE_YEARS', 'EMPLOYED_YEARS',
            'EMPLOYMENT_TO_AGE_RATIO'
        ]
        
        self.categorical_cols = [
            'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
        ]
        
        # Build processing pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numerical_cols),
                ('cat', cat_pipeline, self.categorical_cols)
            ]
        )
        
        # Full data prep pipeline
        self.full_pipeline = Pipeline([
            ('engineer', CreditFeatureEngineer()),
            ('preprocess', self.preprocessor)
        ])
        
        self.fitted_feature_names = []

    def fit(self, df: pd.DataFrame):
        """Fits the preprocessor to the training dataframe."""
        # Split target if present
        X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        
        # Fit full pipeline
        self.full_pipeline.fit(X)
        
        # Extract fitted feature names for tracking importances
        engineer = self.full_pipeline.named_steps['engineer']
        X_engineered = engineer.transform(X)
        
        # Numerical features stay in order
        num_features = self.numerical_cols
        
        # Categorical features are one-hot encoded
        cat_transformer = self.preprocessor.named_transformers_['cat']
        cat_encoder = cat_transformer.named_steps['onehot']
        
        # Handle possible empty or dummy categoricals securely
        if len(self.categorical_cols) > 0:
            cat_features = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        else:
            cat_features = []
            
        self.fitted_feature_names = num_features + cat_features
        logger.info(f"Preprocessor fitted with {len(self.fitted_feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms a dataframe using the fitted pipeline."""
        X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
        return self.full_pipeline.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)
        
    def get_feature_names(self):
        return self.fitted_feature_names
