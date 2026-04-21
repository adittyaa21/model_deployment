"""
Quick Start Guide & Setup Helper
==================================

This script helps set up the project and verify installation.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Requires 3.9+")
        return False


def check_directories():
    """Create required directories."""
    print("\n🔍 Checking directories...")
    
    dirs = ['saved_models', 'mlruns', 'data']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created {dir_name}/")
        else:
            print(f"✅ {dir_name}/ already exists")


def check_data_files():
    """Check if data files exist."""
    print("\n🔍 Checking data files...")
    
    required_files = ['A.csv', 'A_targets.csv']
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} NOT found")
            return False
    
    return True


def install_requirements():
    """Install Python packages."""
    print("\n💾 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing requirements: {str(e)}")
        return False


def test_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('fastapi', 'fastapi'),
        ('streamlit', 'st'),
        ('mlflow', 'mlflow'),
        ('matplotlib', 'plt'),
    ]
    
    all_ok = True
    for package, alias in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT installed")
            all_ok = False
    
    return all_ok


def show_quick_start():
    """Show quick start commands."""
    print_header("QUICK START GUIDE")
    
    print("""
📝 STEP-BY-STEP INSTRUCTIONS:

1️⃣  NOTEBOOKS & EXPLORATION
   Command: jupyter notebook
   File: 01_EDA_and_Modeling.ipynb
   Output: Analysis, visualizations, model comparisons
   
2️⃣  TRAIN MODELS & TRACK EXPERIMENTS
   Command: python 02_ml_pipeline.py
   Output: Models saved in saved_models/
           Experiments logged in mlflow/
   
3️⃣  MONOLITHIC DEPLOYMENT (Option A)
   Command: streamlit run 03_streamlit_app.py
   Access: http://localhost:8501
   
4️⃣  DECOUPLED DEPLOYMENT (Option B)
   Backend:  python 04_fastapi_backend.py
   Frontend: streamlit run 05_streamlit_frontend.py
   
   API Docs: http://localhost:8000/docs
   App: http://localhost:8501

5️⃣  VIEW MLFLOW EXPERIMENTS
   Command: mlflow ui
   Access: http://localhost:5000
    """)


def show_file_structure():
    """Show project file structure."""
    print_header("PROJECT FILE STRUCTURE")
    
    print("""
📁 Project Structure:
    
    ├── 📋 Data
    │   ├── A.csv                          # Features (5000 records, 23 features)
    │   └── A_targets.csv                  # Targets (placement_status, salary_lpa)
    │
    ├── 📓 Notebooks
    │   └── 01_EDA_and_Modeling.ipynb      # Task 1: EDA + 6 Models (25%)
    │
    ├── 🐍 Python Scripts
    │   ├── 02_ml_pipeline.py              # Task 2: Pipeline + MLflow (25%)
    │   ├── 03_streamlit_app.py            # Task 3: Monolithic App (25%)
    │   ├── 04_fastapi_backend.py          # Task 4: FastAPI Backend (20%)
    │   └── 05_streamlit_frontend.py       # Task 4: Streamlit Frontend (20%)
    │
    ├── 📦 Configuration
    │   ├── requirements.txt                # Python dependencies
    │   └── README.md                       # Complete documentation
    │
    ├── 💾 Output Directories
    │   ├── saved_models/                  # Trained models (.pkl files)
    │   ├── mlruns/                        # MLflow artifacts
    │   └── data/                          # Processed data (optional)
    """)


def show_test_commands():
    """Show testing commands."""
    print_header("API TESTING COMMANDS")
    
    print("""
🧪 TEST THE FASTAPI BACKEND:

1. Start Backend:
   python 04_fastapi_backend.py
   
2. Test Classification (Placement):
   curl -X POST "http://localhost:8000/predict/placement" \\
     -H "Content-Type: application/json" \\
     -d '{
       "Student_ID": 1,
       "gender": "Male",
       "branch": "CSE",
       "cgpa": 8.0,
       "tenth_percentage": 80,
       "twelfth_percentage": 85,
       "backlogs": 0,
       "study_hours_per_day": 5.0,
       "attendance_percentage": 85,
       "projects_completed": 8,
       "internships_completed": 3,
       "coding_skill_rating": 5,
       "communication_skill_rating": 4,
       "aptitude_skill_rating": 5,
       "hackathons_participated": 3,
       "certifications_count": 3,
       "sleep_hours": 7.0,
       "stress_level": 3,
       "part_time_job": "No",
       "family_income_level": "High",
       "city_tier": "Tier 1",
       "internet_access": "Yes",
       "extracurricular_involvement": "High"
     }'

3. Test Regression (Salary):
   curl -X POST "http://localhost:8000/predict/salary" \\
     -H "Content-Type: application/json" \\
     -d '{...same data...}'

4. Test Batch Processing:
   curl -X POST "http://localhost:8000/predict/batch" \\
     -F "file=@students.csv" \\
     -F "prediction_type=both"

5. API Documentation:
   Open: http://localhost:8000/docs
    """)


def show_features_summary():
    """Show features summary."""
    print_header("FEATURES SUMMARY")
    
    print("""
📊 CLASSIFICATION MODEL (Placement Prediction)
   Algorithms: Logistic Regression, Random Forest, SVM
   Metrics: Accuracy, Precision, Recall, F1-Score
   Target: Binary (Placed / Not Placed)

💰 REGRESSION MODEL (Salary Estimation)
   Algorithms: Linear Regression, Decision Tree, Gradient Boosting
   Metrics: MAE, RMSE, R² Score
   Target: Continuous (Salary in LPA)

🛠️  FEATURE ENGINEERING
   ✓ Skill Index (avg of coding, communication, aptitude)
   ✓ Academic Score (normalized academic metrics)
   ✓ Activity Index (sum of projects, internships, hackathons, certs)
   ✓ Stress-Sleep Ratio (mental wellness indicator)
   ✓ Engagement Score (study hours + attendance)

🔄 PREPROCESSING
   ✓ StandardScaler for numerical features
   ✓ Label encoding for binary categorical
   ✓ One-hot encoding for multi-class categorical
   ✓ 80:20 train-test split with stratification
    """)


def main():
    """Main setup function."""
    print_header("STUDENT PLACEMENT & SALARY PREDICTION SYSTEM")
    print("Setup & Verification Tool\n")
    
    # Checks
    checks_passed = True
    
    if not check_python_version():
        checks_passed = False
    
    check_directories()
    
    if not check_data_files():
        print("\n⚠️  Data files missing! Please add A.csv and A_targets.csv")
        checks_passed = False
    
    if not test_imports():
        print("\n⚠️  Some packages not installed. Install with:")
        print("   pip install -r requirements.txt")
        checks_passed = False
    
    # Show options
    if checks_passed:
        print("\n✅ Setup verification complete!")
    else:
        print("\n❌ Some checks failed. Please resolve issues above.")
    
    show_file_structure()
    show_features_summary()
    show_quick_start()
    show_test_commands()
    
    print("\n" + "=" * 80)
    print("📚 For more information, see README.md")
    print("=" * 80 + "\n")
    
    return checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
