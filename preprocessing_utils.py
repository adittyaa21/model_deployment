"""
Data Preprocessing Utilities

Consistent preprocessing for training and inference pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import os


class DataPreprocessor:
    """Handles all data preprocessing consistently."""
    
    # Feature names that the model expects (saved after training)
    FEATURE_NAMES_FILE = 'saved_models/feature_names.json'
    
    # All possible categorical values (for one-hot encoding consistency)
    CATEGORICAL_MAPPINGS = {
        'branch': ['CSE', 'ECE', 'IT', 'CE'],
        'city_tier': ['Tier 1', 'Tier 2', 'Tier 3'],
        'family_income_level': ['Low', 'Medium', 'High'],
        'extracurricular_involvement': ['Low', 'Medium', 'High'],
        'gender': ['Male', 'Female'],
        'part_time_job': ['No', 'Yes'],
        'internet_access': ['No', 'Yes']
    }
    
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
    def encode_categorical(df: pd.DataFrame, for_training: bool = False) -> pd.DataFrame:
        """Encode categorical variables consistently."""
        df_enc = df.copy()
        
        binary_mappings = {
            'gender': {'Male': 0, 'Female': 1},
            'part_time_job': {'No': 0, 'Yes': 1},
            'internet_access': {'No': 0, 'Yes': 1}
        }
        
        for col, mapping in binary_mappings.items():
            if col in df_enc.columns:
                df_enc[col] = df_enc[col].map(mapping)
        
        categorical_cols = ['branch', 'city_tier', 'family_income_level', 'extracurricular_involvement']
        
        for col in categorical_cols:
            if col in df_enc.columns:
                dummies = pd.get_dummies(
                    df_enc[col], 
                    prefix=col, 
                    prefix_sep='_',
                    drop_first=True,
                    dtype=int
                )
                
                for category in DataPreprocessor.CATEGORICAL_MAPPINGS.get(col, [])[1:]:
                    col_name = f"{col}_{category}"
                    if col_name not in dummies.columns:
                        dummies[col_name] = 0
                
                df_enc = pd.concat([df_enc, dummies], axis=1)
                df_enc = df_enc.drop(columns=[col])
        
        return df_enc
    
    @staticmethod
    def preprocess(df: pd.DataFrame, drop_targets: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        df_proc = df.copy()
        
        # Feature engineering
        df_proc = DataPreprocessor.engineer_features(df_proc)
        
        # Categorical encoding
        df_proc = DataPreprocessor.encode_categorical(df_proc)
        
        # Drop non-numeric columns except targets
        if drop_targets:
            df_proc = df_proc.drop(columns=['Student_ID', 'placement_status', 'salary_lpa'], errors='ignore')
        else:
            df_proc = df_proc.drop(columns=['Student_ID'], errors='ignore')
        
        # Ensure all columns are numeric
        numeric_cols = [col for col in df_proc.columns if df_proc[col].dtype in ['int64', 'float64']]
        df_proc = df_proc[numeric_cols]
        
        return df_proc
    
    @staticmethod
    def save_feature_names(feature_names: list, filepath: str = None):
        """Save feature names for inference."""
        if filepath is None:
            filepath = DataPreprocessor.FEATURE_NAMES_FILE
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(feature_names, f)
    
    @staticmethod
    def load_feature_names(filepath: str = None) -> list:
        """Load expected feature names for inference."""
        if filepath is None:
            filepath = DataPreprocessor.FEATURE_NAMES_FILE
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def align_features(df: pd.DataFrame, expected_features: list = None) -> pd.DataFrame:
        """Align dataframe features to expected feature set."""
        if expected_features is None:
            expected_features = DataPreprocessor.load_feature_names()
        
        if expected_features is None:
            raise ValueError("No expected features provided or saved")
        
        # Add missing columns with 0
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        
        # Select only expected columns in correct order
        df = df[expected_features]
        
        return df


if __name__ == "__main__":
    print("✓ Data Preprocessor utility loaded successfully")
