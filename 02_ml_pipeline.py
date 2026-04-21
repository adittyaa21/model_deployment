"""
ML Pipeline for Student Placement Prediction

Implements end-to-end pipeline with data ingestion, preprocessing,
model training, and experiment tracking.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import pickle
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')


class DataIngestion:
    """Modular data loading and initial processing."""
    
    def __init__(self, features_path: str, targets_path: str):
        """
        Initialize data ingestion.
        
        Args:
            features_path: Path to features CSV
            targets_path: Path to targets CSV
        """
        self.features_path = features_path
        self.targets_path = targets_path
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge feature and target data."""
        df_features = pd.read_csv(self.features_path)
        df_targets = pd.read_csv(self.targets_path)
        df = df_features.merge(df_targets, on='Student_ID', how='inner')
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df


class FeatureEngineer:
    """Feature engineering and preprocessing."""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw data."""
        df_eng = df.copy()
        
        df_eng['skill_index'] = (
            df_eng['coding_skill_rating'] + 
            df_eng['communication_skill_rating'] + 
            df_eng['aptitude_skill_rating']
        ) / 3
        
        df_eng['academic_score'] = (
            (df_eng['cgpa'] * 10) + 
            df_eng['tenth_percentage'] + 
            df_eng['twelfth_percentage']
        ) / 31
        
        df_eng['activity_index'] = (
            df_eng['projects_completed'] + 
            df_eng['internships_completed'] + 
            df_eng['hackathons_participated'] + 
            df_eng['certifications_count']
        )
        
        df_eng['stress_sleep_ratio'] = df_eng['stress_level'] / (df_eng['sleep_hours'] + 0.1)
        
        df_eng['engagement_score'] = (
            df_eng['study_hours_per_day'] + 
            (df_eng['attendance_percentage'] / 20)
        ) / 2
        
        return df_eng
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame, task: str = 'classification'):
        """Preprocess data for modeling."""
        df_proc = df.copy()
        
        le_gender = LabelEncoder()
        df_proc['gender'] = le_gender.fit_transform(df_proc['gender'])
        
        le_part = LabelEncoder()
        df_proc['part_time_job'] = le_part.fit_transform(df_proc['part_time_job'])
        
        le_internet = LabelEncoder()
        df_proc['internet_access'] = le_internet.fit_transform(df_proc['internet_access'])
        
        df_proc = pd.get_dummies(df_proc, columns=['branch', 'city_tier', 'family_income_level',
                                                    'extracurricular_involvement'],
                                drop_first=True)
        
        if task == 'classification':
            y = (df_proc['placement_status'] == 'Placed').astype(int)
            df_proc = df_proc.drop(columns=['placement_status', 'salary_lpa'])
        else:
            y = df_proc['salary_lpa']
            df_proc = df_proc.drop(columns=['placement_status', 'salary_lpa'])
        
        X = df_proc.drop(columns=['Student_ID'])
        return X, y


class MLPipeline:
    """Unified ML Pipeline with preprocessing and modeling."""
    
    def __init__(self, model_name: str, model, task: str = 'classification'):
        """Initialize pipeline with model and task type."""
        self.model_name = model_name
        self.model = model
        self.task = task
        self.pipeline = None
    
    def build_pipeline(self):
        """Create preprocessing and model pipeline."""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model)
        ])
    
    def train(self, X_train, y_train):
        """Train the pipeline."""
        print(f"Training {self.model_name}...")
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        return self.pipeline.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        if self.task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'f1_score': f1
            }
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            metrics = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse
            }
            print(f"   R² Score: {r2:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   RMSE: {rmse:.4f}")
        
        return metrics, y_pred
    
    def save_model(self, filepath: str):
        """Save model to pickle file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {filepath}")


def run_experiment(task: str = 'classification'):
    """Run complete ML experiment with MLflow tracking."""
    print(f"\nML Pipeline - {task.upper()} TASK")
    
    # Set MLflow tracking
    mlflow.set_experiment(f"Student_Placement_{task.capitalize()}")
    
    # 1. Data Ingestion
    ingestion = DataIngestion('A.csv', 'A_targets.csv')
    df = ingestion.load_data()
    
    # 2. Feature Engineering
    df_engineered = FeatureEngineer.engineer_features(df)
    
    # 3. Data Preprocessing
    X, y = FeatureEngineer.preprocess_data(df_engineered, task)
    
    # 4. Train-Test Split (80:20)
    if task == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Data Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # 5. Model Training and Evaluation
    if task == 'classification':
        models = [
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('SVM', SVC(kernel='rbf', random_state=42))
        ]
    else:
        models = [
            ('Linear Regression', LinearRegression()),
            ('Decision Tree', DecisionTreeRegressor(max_depth=15, random_state=42)),
            ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
    
    best_model = None
    best_score = -np.inf
    best_metrics = None
    
    for model_name, model in models:
        print(f"\nTraining: {model_name}")
        
        with mlflow.start_run(run_name=model_name):
            # Build and train pipeline
            pipeline = MLPipeline(model_name, model, task)
            pipeline.build_pipeline()
            pipeline.train(X_train, y_train)
            
            # Evaluate
            metrics, y_pred = pipeline.evaluate(X_test, y_test)
            
            # Log with MLflow
            mlflow.log_params({
                'task': task,
                'model': model_name,
                'test_size': 0.2,
                'random_state': 42
            })
            mlflow.log_metrics(metrics)
            
            # Save pipeline artifact (use model_name format that MLflow accepts)
            model_artifact_name = f"{task}_{model_name.replace(' ', '_')}"
            mlflow.sklearn.log_model(pipeline.pipeline, model_artifact_name)
            
            # Cross-validation score
            if task == 'classification':
                cv_score = cross_val_score(pipeline.pipeline, X_train, y_train, cv=5, scoring='f1_weighted').mean()
                mlflow.log_metric('cv_f1_score', cv_score)
                score = metrics['f1_score']
                print(f"CV F1-Score: {cv_score:.4f}")
            else:
                cv_score = cross_val_score(pipeline.pipeline, X_train, y_train, cv=5, scoring='r2').mean()
                mlflow.log_metric('cv_r2_score', cv_score)
                score = metrics['r2_score']
                print(f"CV R² Score: {cv_score:.4f}")
            
            # Track best model
            if score > best_score:
                best_score = score
                best_model = pipeline
                best_metrics = metrics
    
    print(f"\nBest Model: {best_model.model_name}")
    print(f"Metrics: {best_metrics}")
    
    # Save to pickle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"saved_models/{task}_{best_model.model_name.replace(' ', '_')}_{timestamp}.pkl"
    best_model.save_model(save_path)
    
    # Save feature names for inference
    os.makedirs('saved_models', exist_ok=True)
    feature_names_path = f"saved_models/feature_names_{task}.json"
    with open(feature_names_path, 'w') as f:
        json.dump(X_train.columns.tolist(), f)
    print(f"💾 Feature names saved to {feature_names_path}")
    print(f"   Total features: {len(X_train.columns)}")
    
    # Log best model
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.log_params({
            'best_model': best_model.model_name,
            'task': task,
            'num_features': len(X_train.columns)
        })
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model.pipeline, f"best_model_{task}")
    
    return best_model, best_metrics, save_path


if __name__ == "__main__":
    clf_model, clf_metrics, clf_path = run_experiment(task='classification')
    print("\nClassification task completed")
    
    reg_model, reg_metrics, reg_path = run_experiment(task='regression')
    print("\nRegression task completed")
    
    print(f"\nModels saved:")
    print(f"  Classification: {clf_path}")
    print(f"  Regression: {reg_path}")
    print(f"  MLflow tracking: {mlflow.get_tracking_uri()}")
