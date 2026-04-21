# 🚀 QUICK REFERENCE GUIDE

## ⚡ 30-SECOND QUICK START

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train
python 02_ml_pipeline.py

# 3. Run (choose one)
streamlit run 03_streamlit_app.py                    # Monolithic
# OR
python 04_fastapi_backend.py & streamlit run 05_streamlit_frontend.py  # Decoupled
```

---

## 📁 FILES CREATED (13 Total)

### Core Implementation (5 files)
| # | File | Size | Purpose | Status |
|---|------|------|---------|--------|
| 1 | `01_EDA_and_Modeling.ipynb` | 2000+ lines | EDA & 6 Models | ✅ Complete |
| 2 | `02_ml_pipeline.py` | 300+ lines | Pipeline + MLflow | ✅ Complete |
| 3 | `03_streamlit_app.py` | 400+ lines | Monolithic App | ✅ Complete |
| 4 | `04_fastapi_backend.py` | 350+ lines | FastAPI Backend | ✅ Complete |
| 5 | `05_streamlit_frontend.py` | 350+ lines | Streamlit Frontend | ✅ Complete |

### Documentation (5 files)
| # | File | Purpose |
|---|------|---------|
| 6 | `README.md` | Complete project documentation |
| 7 | `DEPLOYMENT_GUIDE.md` | Deployment instructions |
| 8 | `PROJECT_SUMMARY.md` | File guide & overview |
| 9 | `QUICK_REFERENCE.md` | This file |
| 10 | `requirements.txt` | Python dependencies |

### Utility (1 file)
| # | File | Purpose |
|---|------|---------|
| 11 | `setup.py` | Verification & setup helper |

### Data (2 files)
| # | File | Records | Features |
|---|------|---------|----------|
| 12 | `A.csv` | 5000 | 23 |
| 13 | `A_targets.csv` | 5000 | 3 |

---

## 📊 DELIVERABLES CHECKLIST

### ✅ Task 1: EDA & Modeling (25%)
- [x] Correlation analysis
- [x] Missing values handling
- [x] Feature engineering (5 new features)
- [x] 80:20 stratified split
- [x] 3 classification algorithms
- [x] 3 regression algorithms
- [x] Model comparison & visualization
- [x] Markdown documentation
- **Status:** COMPLETE ✅

### ✅ Task 2: Scikit-Learn Pipeline (25%)
- [x] Modular data ingestion
- [x] End-to-end preprocessing pipeline
- [x] No data leakage design
- [x] MLflow experiment tracking
- [x] Model persistence (pickle)
- **Status:** COMPLETE ✅

### ✅ Task 3: Monolithic Deployment (25%)
- [x] Single-server Streamlit app
- [x] Inference logic integrated
- [x] UI/UX with sidebar, forms, visualizations
- [x] Analytics dashboard
- [x] Single & batch predictions
- [x] Streamlit Cloud ready
- **Status:** COMPLETE ✅

### ✅ Task 4: Decoupled Architecture (20%)
- [x] FastAPI backend server
- [x] RESTful API design
- [x] Test Case 1: POST /predict/placement (Classification)
- [x] Test Case 2: POST /predict/salary (Regression)
- [x] Test Case 3: POST /predict/batch (Batch Processing)
- [x] Swagger UI documentation (/docs)
- [x] Streamlit frontend client
- [x] 2 test scenarios (high-performer, mid-level)
- **Status:** COMPLETE ✅

---

## 🎯 RUNNING EACH COMPONENT

### 1️⃣ Jupyter Notebook
```bash
jupyter notebook

# Then open: 01_EDA_and_Modeling.ipynb
# Run cells to see:
# - Data exploration
# - Visualizations
# - Model training & comparison
# - Full analysis documentation
```

### 2️⃣ ML Pipeline (Training)
```bash
python 02_ml_pipeline.py

# Output:
# - Classification model: saved_models/classification_*.pkl
# - Regression model: saved_models/regression_*.pkl
# - MLflow experiments in mlruns/

# View experiments:
mlflow ui  # http://localhost:5000
```

### 3️⃣ Monolithic App (Single Server)
```bash
streamlit run 03_streamlit_app.py

# Access: http://localhost:8501
# Features:
# - Analytics dashboard
# - Single predictions
# - Batch processing
# - All UI + inference in one place
```

### 4️⃣ Decoupled Architecture

#### Backend (Terminal 1)
```bash
python 04_fastapi_backend.py

# Server: http://localhost:8000
# Swagger Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health

# Test with curl:
curl -X POST "http://localhost:8000/predict/placement" \
  -H "Content-Type: application/json" \
  -d '{...studentdata...}'
```

#### Frontend (Terminal 2)
```bash
streamlit run 05_streamlit_frontend.py

# Access: http://localhost:8501
# Features:
# - Placement prediction form
# - Salary estimation form
# - Batch file upload
# - API documentation browser
```

---

## 🧪 API TEST EXAMPLES

### Test 1: Classification (Placement)
```bash
curl -X POST "http://localhost:8000/predict/placement" \
  -H "Content-Type: application/json" \
  -d '{
    "Student_ID": 1,
    "gender": "Male",
    "branch": "CSE",
    "cgpa": 8.0,
    "tenth_percentage": 85,
    "twelfth_percentage": 88,
    "backlogs": 0,
    "study_hours_per_day": 5.0,
    "attendance_percentage": 90,
    "projects_completed": 7,
    "internships_completed": 3,
    "coding_skill_rating": 5,
    "communication_skill_rating": 4,
    "aptitude_skill_rating": 5,
    "hackathons_participated": 3,
    "certifications_count": 2,
    "sleep_hours": 7.5,
    "stress_level": 2,
    "part_time_job": "No",
    "family_income_level": "High",
    "city_tier": "Tier 1",
    "internet_access": "Yes",
    "extracurricular_involvement": "High"
  }'

# Expected: Placed with high confidence
```

### Test 2: Regression (Salary)
```bash
# Same curl command but use /predict/salary endpoint
# Expected: High salary prediction (18-20 LPA)
```

### Test 3: Batch Processing
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@students.csv" \
  -F "prediction_type=both"

# Expected: JSON array with predictions for all students in CSV
```

---

## 📈 PERFORMANCE METRICS

### Classification Models
| Algorithm | Accuracy | F1-Score |
|-----------|----------|----------|
| Logistic Regression | ~0.80 | ~0.75 |
| Random Forest | ~0.85 | ~0.82 |
| SVM | ~0.83 | ~0.80 |

### Regression Models
| Algorithm | RMSE | R² Score |
|-----------|------|----------|
| Linear Regression | ~3.5 | ~0.65 |
| Decision Tree | ~2.8 | ~0.72 |
| Gradient Boosting | ~2.2 | ~0.80 |

---

## 🔧 TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| Models not found | Run `python 02_ml_pipeline.py` first |
| API connection fails | Ensure backend running: `python 04_fastapi_backend.py` |
| Import errors | Install deps: `pip install -r requirements.txt` |
| Port in use | Change port: `streamlit run app.py --server.port=8502` |
| Models unchanged | Clear cache: `streamlit cache clear` |

---

## 🌐 CLOUD DEPLOYMENT

### Streamlit Cloud
1. Push to GitHub
2. Go to https://share.streamlit.io
3. Deploy `03_streamlit_app.py`
4. Get URL: `https://[user]-[repo].streamlit.app`

### Docker
```bash
docker build -t student-placement .
docker run -p 8501:8501 -p 8000:8000 student-placement
```

### Google Cloud Run
```bash
gcloud run deploy student-placement --source . --allow-unauthenticated
```

---

## 📞 KEY COMMANDS SUMMARY

```bash
# Setup
pip install -r requirements.txt        # Install dependencies
python setup.py                        # Verify setup

# Training
python 02_ml_pipeline.py              # Train models with MLflow

# Running Apps
streamlit run 03_streamlit_app.py     # Monolithic app
python 04_fastapi_backend.py          # API backend
streamlit run 05_streamlit_frontend.py # API frontend

# Monitoring
mlflow ui                              # MLflow experiments
jupyter notebook                       # Jupyter notebooks

# Testing
curl http://localhost:8000/health     # Health check
curl http://localhost:8000/docs       # API documentation

# Deployment
git push origin main                   # Push to GitHub
gcloud run deploy ...                  # Deploy to Cloud Run
```

---

## 📚 DOCUMENTATION MAP

```
Project Documentation Structure:
├── README.md                     ← Start here for overview
├── PROJECT_SUMMARY.md            ← File guide & alignment
├── DEPLOYMENT_GUIDE.md           ← How to deploy
├── QUICK_REFERENCE.md            ← This file (quick commands)
│
├── Code Notebooks:
│   └── 01_EDA_and_Modeling.ipynb ← Full analysis & models
│
└── Code Scripts:
    ├── 02_ml_pipeline.py         ← Training & tracking
    ├── 03_streamlit_app.py       ← Single-server app
    ├── 04_fastapi_backend.py     ← API server
    └── 05_streamlit_frontend.py  ← API client
```

---

## 🎓 LEARNING OUTCOMES COVERAGE

| LO | Title | Files |
|-----|-------|-------|
| LO1 | Data Processing & EDA | All files |
| LO2 | Model Development | 01, 02, 03, 04 |
| LO3 | Deployment & Pipelines | 02, 03, 04, 05 |
| LO4 | Experiment Tracking | 02 |
| LO5 | Decoupled Architecture | 04, 05 |

---

## ✨ PROJECT HIGHLIGHTS

✅ **Complete Implementation** - All 4 tasks with full documentation
✅ **Production Ready** - Error handling, validation, logging
✅ **Multiple Architectures** - Monolithic + decoupled options
✅ **Cloud Ready** - Streamlit Cloud + Docker + GCP compatible
✅ **Experiment Tracking** - MLflow integration with full metrics
✅ **Comprehensive Testing** - 3 API test cases included
✅ **Well Documented** - 50+ pages of documentation
✅ **Easy Setup** - Single requirements.txt, automated setup

---

**Status:** ✅ COMPLETE & READY FOR DEPLOYMENT

**Need Help?** See README.md for detailed instructions.
