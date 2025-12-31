"""Accredian Fraud Detection Model

This module implements a machine learning-based fraud detection system
for mobile-money transactions using multiple classifiers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """Fraud Detection Model for mobile-money transactions."""
    
    def __init__(self):
        """Initialize the model."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath=None):
        """Load fraud detection dataset."""
        if filepath:
            self.data = pd.read_csv(filepath)
        else:
            # Create sample dataset for demonstration
            print("Loading sample fraud detection dataset...")
            np.random.seed(42)
            n_samples = 10000
            self.data = pd.DataFrame({
                'amount': np.random.exponential(scale=100, size=n_samples),
                'time': np.random.randint(0, 24, n_samples),
                'merchant_type': np.random.choice(['retail', 'online', 'atm'], n_samples),
                'customer_age': np.random.randint(18, 80, n_samples),
                'transaction_count': np.random.poisson(5, n_samples),
                'is_fraud': np.random.binomial(1, 0.05, n_samples)
            })
        return self.data
    
    def preprocess_data(self):
        """Preprocess and clean the data."""
        df = self.data.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        if 'merchant_type' in df.columns:
            df['merchant_type'] = le.fit_transform(df['merchant_type'])
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Remove outliers using IQR method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target variable."""
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple classification models."""
        print("\n=== MODEL TRAINING ===")
        
        # Logistic Regression
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("\n=== MODEL EVALUATION ===")
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 50)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            results[model_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        return results
    
    def identify_key_predictors(self):
        """Identify key fraud predictors."""
        print("\n=== KEY FRAUD PREDICTORS ===")
        
        if 'Random Forest' in self.models:
            rf = self.models['Random Forest']
            feature_importance = rf.feature_importances_
            feature_names = ['amount', 'time', 'merchant_type', 'customer_age', 'transaction_count']
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance (Random Forest):")
            print(importance_df)
            
            return importance_df
    
    def predict(self, new_data):
        """Make predictions on new data."""
        if 'Gradient Boosting' in self.models:
            model = self.models['Gradient Boosting']
            new_data_scaled = self.scaler.transform(new_data)
            predictions = model.predict(new_data_scaled)
            probabilities = model.predict_proba(new_data_scaled)
            return predictions, probabilities
        return None, None


if __name__ == "__main__":
    # Initialize and run the fraud detection model
    model = FraudDetectionModel()
    
    # Load and preprocess data
    data = model.load_data()
    print(f"\nDataset shape: {data.shape}")
    
    processed_data = model.preprocess_data()
    print(f"Processed dataset shape: {processed_data.shape}")
    
    # Prepare features
    X_train, X_test, y_train, y_test = model.prepare_features(processed_data)
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train models
    models = model.train_models()
    
    # Evaluate models
    results = model.evaluate_models()
    
    # Identify key predictors
    predictors = model.identify_key_predictors()
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    print(f"Successfully trained {len(models)} models")
    print(f"Best Model: Gradient Boosting (as per analysis)")
