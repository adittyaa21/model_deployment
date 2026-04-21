"""
Test Script - Validation

Verifies system configuration and data pipeline.
"""

import os
import json
import pandas as pd
import sys
from pathlib import Path

def print_step(step_num, step_name):
    """Print step header."""
    print(f"\nSTEP {step_num}: {step_name}")

def check_files_exist():
    """Check if required files exist."""
    print_step(1, "Checking Required Files")
    
    required_files = {
        'Data': ['A.csv', 'A_targets.csv'],
        'Python Scripts': ['02_ml_pipeline.py', '04_fastapi_backend.py'],
        'Documentation': ['README.md', 'FIX_FEATURE_MISMATCH.md']
    }
    
    all_good = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            if os.path.exists(file):
                print(f"  OK: {file}")
            else:
                print(f"  MISSING: {file}")
                all_good = False
    
    return all_good

def check_models_saved():
    """Check if models and feature names are saved."""
    print_step(2, "Checking Saved Models & Features")
    
    model_dir = 'saved_models'
    
    if not os.path.exists(model_dir):
        print(f"ERROR: {model_dir}/ directory not found")
        print("Run: python 02_ml_pipeline.py")
        return False
    
    print(f"OK: {model_dir}/ exists")
    
    files = os.listdir(model_dir)
    
    print(f"\nFiles in {model_dir}/ ({len(files)} total):")
    
    clf_model = None
    reg_model = None
    clf_features = False
    reg_features = False
    
    for file in sorted(files):
        filepath = os.path.join(model_dir, file)
        size = os.path.getsize(filepath) / 1024  # KB
        
        if 'classification' in file and file.endswith('.pkl'):
            print(f"  OK: {file} ({size:.1f} KB)")
            clf_model = file
        elif 'regression' in file and file.endswith('.pkl'):
            print(f"  OK: {file} ({size:.1f} KB)")
            reg_model = file
        elif 'feature_names_classification.json' in file:
            print(f"  OK: {file}")
            clf_features = True
        elif 'feature_names_regression.json' in file:
            print(f"  OK: {file}")
            reg_features = True
        else:
            print(f"  INFO: {file}")
    
    if not (clf_model and reg_model):
        print("\nERROR: Missing model files")
        print("Run: python 02_ml_pipeline.py")
        return False
    
    if not (clf_features and reg_features):
        print("\nERROR: Missing feature names files")
        print("Run: python 02_ml_pipeline.py")
        return False
    
    return True

def check_feature_names_content():
    """Check feature names files content."""
    print_step(3, "Checking Feature Names Content")
    
    clf_file = 'saved_models/feature_names_classification.json'
    reg_file = 'saved_models/feature_names_regression.json'
    
    all_good = True
    
    for file_path, task_name in [(clf_file, 'Classification'), (reg_file, 'Regression')]:
        if not os.path.exists(file_path):
            print(f"❌ {task_name}: {file_path} not found")
            all_good = False
            continue
        
        try:
            with open(file_path, 'r') as f:
                features = json.load(f)
            
            print(f"\nOK: {task_name}")
            print(f"  Total features: {len(features)}")
            
            # Check for engineered features
            engineered_features = ['skill_index', 'academic_score', 'activity_index', 
                                  'stress_sleep_ratio', 'engagement_score']
            present_engineered = [f for f in engineered_features if f in features]
            
            print(f"  Engineered features: {len(present_engineered)}/5")
            for feat in present_engineered:
                print(f"    {feat}")
            
            # Check for one-hot encoded features
            onehot_features = [f for f in features if '_' in f and any(
                cat in f for cat in ['branch_', 'city_tier_', 'family_income_level_', 'extracurricular_']
            )]
            
            print(f"  One-hot encoded features: {len(onehot_features)}")
            for feat in onehot_features[:3]:
                print(f"    {feat}")
            if len(onehot_features) > 3:
                print(f"    ... and {len(onehot_features)-3} more")
            
            # Validation: must have engineered + one-hot features
            if len(present_engineered) < 5:
                print(f"  ERROR: Missing engineered features: {set(engineered_features) - set(present_engineered)}")
                all_good = False
            
            if len(onehot_features) < 8:
                print(f"  WARNING: Only {len(onehot_features)} one-hot features (expected ~9-10)")
                # Don't fail on this, just warn
        
        except Exception as e:
            print(f"ERROR reading {file_path}: {str(e)}")
            all_good = False
    
    return all_good

def check_data_files():
    """Check data files."""
    print_step(4, "Checking Data Files")
    
    try:
        df_features = pd.read_csv('A.csv')
        df_targets = pd.read_csv('A_targets.csv')
        
        print(f"OK: Features data")
        print(f"  Shape: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns")
        
        print(f"\nOK: Targets data")
        print(f"  Shape: {df_targets.shape[0]:,} rows × {df_targets.shape[1]} columns")
        
        print(f"\nOK: Data ready for merge on Student_ID")
        
        return True
    
    except Exception as e:
        print(f"ERROR reading data files: {str(e)}")
        return False

def check_preprocessing_utils():
    """Check if preprocessing_utils module exists."""
    print_step(5, "Checking Preprocessing Utilities")
    
    if os.path.exists('preprocessing_utils.py'):
        print("OK: preprocessing_utils.py exists")
        return True
    else:
        print("ERROR: preprocessing_utils.py not found")
        return False

def simulate_preprocessing():
    """Simulate preprocessing to check for errors."""
    print_step(6, "Simulating Preprocessing")
    
    try:
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        # Create sample student data
        sample_student = {
            'Student_ID': 1,
            'gender': 'Male',
            'branch': 'CSE',
            'cgpa': 8.0,
            'tenth_percentage': 85,
            'twelfth_percentage': 88,
            'backlogs': 0,
            'study_hours_per_day': 5.0,
            'attendance_percentage': 90,
            'projects_completed': 7,
            'internships_completed': 3,
            'coding_skill_rating': 5,
            'communication_skill_rating': 4,
            'aptitude_skill_rating': 5,
            'hackathons_participated': 3,
            'certifications_count': 2,
            'sleep_hours': 7.5,
            'stress_level': 2,
            'part_time_job': 'No',
            'family_income_level': 'High',
            'city_tier': 'Tier 1',
            'internet_access': 'Yes',
            'extracurricular_involvement': 'High'
        }
        
        df = pd.DataFrame([sample_student])
        
        # Apply feature engineering
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
        
        print("OK: Feature engineering works")
        print(f"  skill_index: {df['skill_index'].values[0]:.2f}")
        print(f"  academic_score: {df['academic_score'].values[0]:.2f}")
        print(f"  activity_index: {df['activity_index'].values[0]:.1f}")
        print(f"  stress_sleep_ratio: {df['stress_sleep_ratio'].values[0]:.2f}")
        print(f"  engagement_score: {df['engagement_score'].values[0]:.2f}")
        
        # Apply encoding
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
        
        print("\nOK: Categorical encoding works")
        print(f"  Total columns after encoding: {df.shape[1]}")
        print(f"  Numeric columns: {len([c for c in df.columns if df[c].dtype in ['int64', 'float64']])}")
        
        return True
    
    except Exception as e:
        print(f"ERROR during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def final_recommendations():
    """Print final recommendations."""
    print_step(7, "Final Recommendations")
    
    print("""
Next Steps:

1. If models exist:
  Ready to use
  Backend: python 04_fastapi_backend.py
  Frontend: streamlit run 05_streamlit_frontend.py
  Access at http://localhost:8501

2. If models missing:
  Train first: python 02_ml_pipeline.py
  Then run backend and frontend

3. Test API:
  python 04_fastapi_backend.py
  Open http://localhost:8000/docs

4. Test monolithic app:
  streamlit run 03_streamlit_app.py
  Open http://localhost:8501
    """)

def main():
    """Run validation checks."""
    print("\nVALIDATION TEST")
    
    checks = [
        ("Files Exist", check_files_exist),
        ("Models Saved", check_models_saved),
        ("Feature Names Content", check_feature_names_content),
        ("Data Files", check_data_files),
        ("Preprocessing Utils", check_preprocessing_utils),
        ("Preprocessing Simulation", simulate_preprocessing),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\nError in {check_name}: {str(e)}")
            results.append((check_name, False))
    
    print("\nTEST SUMMARY")
    
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {check_name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\nAll checks passed. System ready.")
    else:
        print("\nSome checks failed. See recommendations above.")
    
    final_recommendations()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
