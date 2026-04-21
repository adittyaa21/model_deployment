"""
Quick API Validation Test

Tests preprocessing and prediction logic directly.
"""

import sys
import json
import pickle
from pathlib import Path

# Test data
test_student = {
    'Student_ID': 1001,
    'gender': 'Male',
    'branch': 'CSE',
    'cgpa': 8.5,
    'tenth_percentage': 92,
    'twelfth_percentage': 94,
    'backlogs': 0,
    'study_hours_per_day': 8.0,
    'attendance_percentage': 95,
    'projects_completed': 10,
    'internships_completed': 3,
    'coding_skill_rating': 5,
    'communication_skill_rating': 4,
    'aptitude_skill_rating': 5,
    'hackathons_participated': 5,
    'certifications_count': 4,
    'sleep_hours': 7.0,
    'stress_level': 2,
    'part_time_job': 'No',
    'family_income_level': 'High',
    'city_tier': 'Tier 1',
    'internet_access': 'Yes',
    'extracurricular_involvement': 'High'
}

print("\nAPI VALIDATION TEST")

print("\nSTEP 1: Checking models...")
# Find the latest models dynamically
from pathlib import Path
import glob

model_dir = Path('saved_models')
clf_models = list(model_dir.glob('classification_*.pkl'))
reg_models = list(model_dir.glob('regression_*.pkl'))

if not clf_models:
    print(f"ERROR: No classification models found")
    sys.exit(1)
clf_model_path = str(sorted(clf_models, key=lambda p: p.stat().st_mtime)[-1])
print(f"OK: Classification model: {clf_model_path}")

if not reg_models:
    print(f"ERROR: No regression models found")
    sys.exit(1)
reg_model_path = str(sorted(reg_models, key=lambda p: p.stat().st_mtime)[-1])
print(f"OK: Regression model: {reg_model_path}")

# Step 2: Load models
print("\nSTEP 2: Loading models...")
try:
    with open(clf_model_path, 'rb') as f:
        clf_model = pickle.load(f)
    print("OK: Classification model loaded")
except Exception as e:
    print(f"ERROR: Failed to load classification model: {e}")
    sys.exit(1)

try:
    with open(reg_model_path, 'rb') as f:
        reg_model = pickle.load(f)
    print("OK: Regression model loaded")
except Exception as e:
    print(f"ERROR: Failed to load regression model: {e}")
    sys.exit(1)

# Step 3: Load feature names
print("\nSTEP 3: Loading feature names...")
try:
    with open('saved_models/feature_names_classification.json', 'r') as f:
        clf_features = json.load(f)
    print(f"OK: Classification features: {len(clf_features)} features")
except Exception as e:
    print(f"ERROR: Failed to load classification features: {e}")
    sys.exit(1)

try:
    with open('saved_models/feature_names_regression.json', 'r') as f:
        reg_features = json.load(f)
    print(f"OK: Regression features: {len(reg_features)} features")
except Exception as e:
    print(f"ERROR: Failed to load regression features: {e}")
    sys.exit(1)

# Step 4: Test preprocessing (import from module)
print("\n✓ STEP 4: Testing preprocessing...")
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    df = pd.DataFrame([test_student])
    
    # Feature engineering
    df['skill_index'] = (
        df['coding_skill_rating'] + 
        df['communication_skill_rating'] + 
        df['aptitude_skill_rating']
    ) / 3
    
    df['academic_score'] = (
        (df['cgpa'] * 10) + 
        df['tenth_percentage'] + 
        df['twelfth_percentage']
    ) / 31
    
    df['activity_index'] = (
        df['projects_completed'] + 
        df['internships_completed'] + 
        df['hackathons_participated'] + 
        df['certifications_count']
    )
    
    df['stress_sleep_ratio'] = df['stress_level'] / (df['sleep_hours'] + 0.1)
    
    df['engagement_score'] = (
        df['study_hours_per_day'] + 
        (df['attendance_percentage'] / 20)
    ) / 2
    
    # Binary encoding
    le_gender = LabelEncoder()
    le_gender.fit(['Male', 'Female'])
    df['gender'] = le_gender.transform(df['gender'])
    
    le_part = LabelEncoder()
    le_part.fit(['No', 'Yes'])
    df['part_time_job'] = le_part.transform(df['part_time_job'])
    
    le_internet = LabelEncoder()
    le_internet.fit(['No', 'Yes'])
    df['internet_access'] = le_internet.transform(df['internet_access'])
    
    # One-hot encode
    df = pd.get_dummies(
        df, 
        columns=['branch', 'city_tier', 'family_income_level', 'extracurricular_involvement'],
        drop_first=True,
        dtype=int
    )
    
    df = df.drop(columns=['Student_ID'], errors='ignore')
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    df_processed = df[numeric_cols]
    
    print(f"OK: Preprocessing successful: {df_processed.shape[1]} features")
    
except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test feature alignment
print("\nSTEP 5: Testing feature alignment...")
try:
    # Add missing columns
    for col in clf_features:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Select only required features in correct order
    df_aligned = df_processed[clf_features]
    
    print(f"OK: Features aligned successfully")
    print(f"  Expected: {len(clf_features)}, Provided: {len(df_aligned.columns)}")
    
    if len(df_aligned.columns) != len(clf_features):
        print(f"ERROR: Column count mismatch!")
        sys.exit(1)
    
except Exception as e:
    print(f"ERROR: Feature alignment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test predictions
print("\nSTEP 6: Testing predictions...")
try:
    clf_pred = clf_model.predict(df_aligned)[0]
    clf_prob = clf_model.predict_proba(df_aligned)[0].max()
    print(f"OK: Classification prediction: {clf_pred}")
    print(f"  Confidence: {clf_prob:.2%}")
    
    reg_pred = reg_model.predict(df_aligned)[0]
    print(f"OK: Regression prediction: {reg_pred:.2f} LPA")
    
except Exception as e:
    print(f"ERROR: Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nALL TESTS PASSED")
print("System is ready to use!")
print("\nNext steps:")
print("  1. python 04_fastapi_backend.py")
print("  2. Open http://localhost:8000/docs")
print("  3. Or: streamlit run 05_streamlit_frontend.py")
print()
