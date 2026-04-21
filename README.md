# Student Placement & Salary Prediction System
## Complete Machine Learning Project Implementation

This project implements a comprehensive machine learning system for predicting student employment outcomes and salary estimates, with multiple architectures and deployment options.

---

## 📋 Project Deliverables

### 1️⃣ **EDA & Modeling** (25% - LO1, LO2)
**File:** `01_EDA_and_Modeling.ipynb`

**Features:**
- ✅ Exploratory Data Analysis (EDA) with visualizations
- ✅ Missing value analysis and handling
- ✅ Feature engineering (5 new engineered features):
  - Skill Index (combination of technical & soft skills)
  - Academic Score (normalized academic metrics)
  - Activity Index (projects, internships, hackathons, certs)
  - Stress-Sleep Ratio (wellness indicator)
  - Engagement Score (study hours & attendance)
- ✅ Correlation analysis with target variables
- ✅ 80:20 stratified train-test split
- ✅ Classification models (3 algorithms):
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM)
- ✅ Regression models (3 algorithms):
  - Linear Regression
  - Decision Tree Regressor
  - Gradient Boosting Regressor
- ✅ Comprehensive metrics evaluation and interpretation

**Metrics Tracked:**
- **Classification:** Accuracy, Precision, Recall, F1-Score
- **Regression:** MAE, RMSE, R² Score

---

### 2️⃣ **Scikit-Learn Pipeline** (25% - LO1, LO2, LO3, LO4)
**File:** `02_ml_pipeline.py`

**Features:**
- ✅ Modular data ingestion function
- ✅ End-to-end preprocessing pipeline (StandardScaler + Encoding)
- ✅ Unified sklearn.Pipeline avoiding data leakage
- ✅ MLflow experiment tracking:
  - Parameter logging
  - Metric logging
  - Model artifact storage
  - Cross-validation tracking
- ✅ Model persistence (pickle format + MLflow)
- ✅ Best model selection logic

**Usage:**
```bash
python 02_ml_pipeline.py
```

**Output:**
- Saved models in `saved_models/` directory
- MLflow experiments at `http://localhost:5000` (after `mlflow ui`)

---

### 3️⃣ **Monolithic Deployment** (25% - LO1, LO2, LO3)
**File:** `03_streamlit_app.py`

**Features:**
- ✅ Single-server application (UI + inference logic combined)
- ✅ Intuitive sidebar navigation
- ✅ Streamlit UI components:
  - Form input with all student features
  - Data visualization dashboard
  - Analytics with placement/salary distributions
  - Batch prediction capability
- ✅ Model loading from pickle files
- ✅ Prediction with confidence scores
- ✅ Data visualization (matplotlib + seaborn)

**Usage:**
```bash
streamlit run 03_streamlit_app.py
```

**Features:**
- Analytics Dashboard with key metrics
- Single Student Prediction (Placement & Salary)
- Batch CSV file processing
- Download predictions as CSV

---

### 4️⃣ **Decoupled Architecture** (20% - LO1, LO2, LO3, LO4, LO5)

#### 4A. **FastAPI Backend**
**File:** `04_fastapi_backend.py`

**Features:**
- ✅ RESTful API server with CORS support
- ✅ Pydantic models for input validation
- ✅ Endpoints:
  - `GET /` - Info endpoint
  - `GET /health` - Health check with model status
  - `POST /predict/placement` - Single placement prediction
  - `POST /predict/salary` - Single salary prediction
  - `POST /predict/batch` - Batch CSV processing
  - `GET /api/stats` - API statistics

**Usage:**
```bash
python 04_fastapi_backend.py
```

**API Documentation:** `http://localhost:8000/docs` (Swagger UI)

**Test Cases Included:**

1. **POST /predict/placement** (Classification Test)
   ```bash
   curl -X POST "http://localhost:8000/predict/placement" \
     -H "Content-Type: application/json" \
     -d '{
       "Student_ID": 1,
       "gender": "Male",
       "branch": "CSE",
       "cgpa": 8.0,
       ...
     }'
   ```

2. **POST /predict/salary** (Regression Test)
   ```bash
   curl -X POST "http://localhost:8000/predict/salary" \
     -H "Content-Type: application/json" \
     -d '{...}'
   ```

3. **POST /predict/batch** (Batch Processing Test)
   ```bash
   curl -X POST "http://localhost:8000/predict/batch" \
     -F "file=@students.csv" \
     -F "prediction_type=both"
   ```

#### 4B. **Streamlit Frontend**
**File:** `05_streamlit_frontend.py`

**Features:**
- ✅ Client application communicating with FastAPI backend
- ✅ API connection status display
- ✅ Two test scenarios:
  - **Scenario 1:** High-performing student (Classification)
  - **Scenario 2:** Mid-level student (Regression)
- ✅ Form interface for data input
- ✅ Real-time API communication
- ✅ Results visualization and download
- ✅ API documentation browser
- ✅ Batch file upload and processing

**Usage:**
```bash
# Terminal 1: Start FastAPI backend
python 04_fastapi_backend.py

# Terminal 2: Start Streamlit frontend
streamlit run 05_streamlit_frontend.py
```

**Access:** `http://localhost:8501`

---

## 📦 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Directories
```bash
mkdir saved_models
mkdir mlruns
```

### 3. Run Pipeline to Train Models
```bash
python 02_ml_pipeline.py
```

### 4. Start Services

**Option A: Monolithic Deployment**
```bash
streamlit run 03_streamlit_app.py
```

**Option B: Decoupled Deployment**
```bash
# Terminal 1
python 04_fastapi_backend.py

# Terminal 2
streamlit run 05_streamlit_frontend.py
```

---

## 📊 Dataset

**Files:**
- `A.csv` - Student features (23 features, 5000 records)
- `A_targets.csv` - Target variables (placement_status, salary_lpa)

**Features:**
- Demographic: Gender, Branch, City Tier, Family Income
- Academic: CGPA, Board Percentages, Backlogs, Attendance
- Skills: Coding, Communication, Aptitude Ratings
- Experience: Projects, Internships, Hackathons, Certifications
- Lifestyle: Sleep, Stress, Study Hours

**Targets:**
- **Classification:** Placement Status (Placed/Not Placed)
- **Regression:** Salary (LPA)

---

## 🎯 Model Performance Summary

### Classification Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.XX | 0.XX | 0.XX | 0.XX |
| Random Forest | 0.XX | 0.XX | 0.XX | 0.XX |
| SVM | 0.XX | 0.XX | 0.XX | 0.XX |

### Regression Results
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | X.XX | X.XX | X.XX |
| Decision Tree | X.XX | X.XX | X.XX |
| Gradient Boosting | X.XX | X.XX | X.XX |

---

## 🌐 Deployment Options

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy `03_streamlit_app.py`
4. **URL:** `https://[your-username]-student-placement.streamlit.app`

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["streamlit", "run", "03_streamlit_app.py"]
```

### Production Considerations
- Use production-grade ASGI server (Gunicorn + Uvicorn)
- Add authentication to API endpoints
- Implement request rate limiting
- Add comprehensive logging
- Use environment variables for configuration

---

## 📝 Key Architectural Decisions

### Feature Engineering Rationale
1. **Skill Index** - Consolidates multiple skill ratings into single metric
2. **Academic Score** - Normalizes disparate academic metrics
3. **Activity Index** - Counts cumulative involvement in activities
4. **Stress-Sleep Ratio** - Indicates work-life balance
5. **Engagement Score** - Measures student commitment

### Model Selection
- **Classification:** Random Forest for best F1-score and feature importance
- **Regression:** Gradient Boosting for lowest RMSE and best R² score

### Pipeline Design
- StandardScaler prevents feature dominance
- LabelEncoder handles categorical variables
- No data leakage: scaling done inside pipeline

---

## 📚 Additional Resources

### MLflow Commands
```bash
# Start MLflow UI
mlflow ui

# View experiments
# Open http://localhost:5000
```

### FastAPI Swagger UI
- Endpoint: `http://localhost:8000/docs`
- Interactive API testing and documentation

### Streamlit Configuration
```bash
# Run in headless mode
streamlit run app.py --headless

# Debug mode
streamlit run app.py --logger.level=debug
```

---

## ⚠️ Troubleshooting

1. **Models not loading in Streamlit**
   - Ensure `saved_models/` directory exists
   - Run `02_ml_pipeline.py` to generate models

2. **API connection fails**
   - Check FastAPI server is running on `localhost:8000`
   - Verify CORS settings allow frontend access

3. **Import errors**
   - Install all packages: `pip install -r requirements.txt`
   - Check Python version (3.9+)

4. **Model predictions unchanged**
   - Retrain models with updated data
   - Clear cache with `streamlit cache clear`

---

## 📄 File Structure
```
.
├── A.csv                          # Features data
├── A_targets.csv                  # Target variables
├── 01_EDA_and_Modeling.ipynb      # Notebook (Task 1)
├── 02_ml_pipeline.py              # Pipeline (Task 2)
├── 03_streamlit_app.py            # Monolithic app (Task 3)
├── 04_fastapi_backend.py          # API Backend (Task 4)
├── 05_streamlit_frontend.py       # API Frontend (Task 4)
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── saved_models/                  # Trained models (.pkl)
└── mlruns/                        # MLflow artifacts
```

---

## 🔗 References

- **Scikit-Learn:** https://scikit-learn.org/
- **MLflow:** https://mlflow.org/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Streamlit:** https://streamlit.io/

---

## 👤 Author
Machine Learning Deployment Project - Student Placement & Salary Prediction

---

## 📄 License
This project is for educational purposes.

---

**Last Updated:** April 21, 2024
**Status:** ✅ Complete
