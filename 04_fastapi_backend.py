"""
FastAPI Backend for Placement Prediction

Provides RESTful API endpoints for model predictions.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from datetime import datetime
import io
import logging
import json
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Student Placement API",
    description="API for student placement and salary predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class StudentData(BaseModel):
    """Student input data model."""
    Student_ID: int = Field(..., example=1)
    gender: str = Field(..., example="Male")
    branch: str = Field(..., example="CSE")
    cgpa: float = Field(..., ge=0, le=10, example=7.5)
    tenth_percentage: float = Field(..., ge=0, le=100, example=75.0)
    twelfth_percentage: float = Field(..., ge=0, le=100, example=75.0)
    backlogs: int = Field(..., ge=0, example=0)
    study_hours_per_day: float = Field(..., ge=0, le=24, example=4.0)
    attendance_percentage: float = Field(..., ge=0, le=100, example=75.0)
    projects_completed: int = Field(..., ge=0, example=5)
    internships_completed: int = Field(..., ge=0, example=2)
    coding_skill_rating: int = Field(..., ge=1, le=5, example=3)
    communication_skill_rating: int = Field(..., ge=1, le=5, example=3)
    aptitude_skill_rating: int = Field(..., ge=1, le=5, example=3)
    hackathons_participated: int = Field(..., ge=0, example=2)
    certifications_count: int = Field(..., ge=0, example=2)
    sleep_hours: float = Field(..., ge=0, le=24, example=7.0)
    stress_level: int = Field(..., ge=1, le=10, example=5)
    part_time_job: str = Field(..., example="No")
    family_income_level: str = Field(..., example="Medium")
    city_tier: str = Field(..., example="Tier 1")
    internet_access: str = Field(..., example="Yes")
    extracurricular_involvement: str = Field(..., example="Medium")


class PredictionResponse(BaseModel):
    """Prediction response model."""
    student_id: int
    prediction_type: str
    prediction: str | float
    confidence: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    total_records: int
    successful: int
    failed: int
    results: List[dict]


class ModelRegistry:
    """Manage model loading and predictions."""
    
    def __init__(self):
        self.classification_model = None
        self.regression_model = None
        self.classification_features = None
        self.regression_features = None
        self.scaler = StandardScaler()
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models from pickle files."""
        try:
            model_dir = 'saved_models'
            
            if not os.path.exists(model_dir):
                logger.warning(f"{model_dir}/ directory not found")
                return
            
            def load_latest_compatible(files, model_type):
                """Try newest-to-oldest model files and return first compatible model."""
                candidates = sorted(
                    files,
                    key=lambda name: os.path.getmtime(os.path.join(model_dir, name)),
                    reverse=True
                )

                for file_name in candidates:
                    file_path = os.path.join(model_dir, file_name)
                    try:
                        with open(file_path, 'rb') as f:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', InconsistentVersionWarning)
                                loaded_model = pickle.load(f)
                        logger.info(f"{model_type} model loaded: {file_name}")
                        return loaded_model
                    except Exception as model_error:
                        logger.warning(
                            f"Skipping incompatible {model_type.lower()} model {file_name}: {model_error}"
                        )

                return None

            # Load classification model
            clf_files = [f for f in os.listdir(model_dir) if 'classification' in f.lower() and f.endswith('.pkl')]
            if clf_files:
                self.classification_model = load_latest_compatible(clf_files, 'Classification')
                if self.classification_model is None:
                    logger.warning("No compatible classification model found")
                
                clf_features_file = os.path.join(model_dir, 'feature_names_classification.json')
                if os.path.exists(clf_features_file):
                    with open(clf_features_file, 'r') as f:
                        self.classification_features = json.load(f)
                    logger.info(f"Classification features loaded: {len(self.classification_features)} features")
            else:
                logger.warning("No classification model found")
            
            # Load regression model
            reg_files = [f for f in os.listdir(model_dir) if 'regression' in f.lower() and f.endswith('.pkl')]
            if reg_files:
                self.regression_model = load_latest_compatible(reg_files, 'Regression')
                if self.regression_model is None:
                    logger.warning("No compatible regression model found")
                
                reg_features_file = os.path.join(model_dir, 'feature_names_regression.json')
                if os.path.exists(reg_features_file):
                    with open(reg_features_file, 'r') as f:
                        self.regression_features = json.load(f)
                    logger.info(f"Regression features loaded: {len(self.regression_features)} features")
            else:
                logger.warning("No regression model found")
        
        except Exception as e:
            error_text = str(e)
            logger.error(f"Error loading models: {error_text}")
            if 'numpy._core' in error_text or 'numpy.core' in error_text:
                logger.error(
                    "Model binary is incompatible with current NumPy/Scikit-learn versions. "
                    "Install dependency versions matching the training environment and restart."
                )
            import traceback
            traceback.print_exc()
            logger.warning("Models can be trained using the ml_pipeline.py script")
    
    @staticmethod
    def preprocess_input(student_data: StudentData) -> pd.DataFrame:
        """Convert input data to model-compatible format with feature engineering."""
        # Convert to dict
        data_dict = student_data.dict()
        df = pd.DataFrame([data_dict])
        
        # ============ FEATURE ENGINEERING ============
        # 1. Skill Index: Average of technical and soft skills
        df['skill_index'] = (
            df['coding_skill_rating'] + 
            df['communication_skill_rating'] + 
            df['aptitude_skill_rating']
        ) / 3
        
        # 2. Academic Score: Normalized academic metrics
        df['academic_score'] = (
            (df['cgpa'] * 10) + 
            df['tenth_percentage'] + 
            df['twelfth_percentage']
        ) / 31
        
        # 3. Activity Index: Sum of activities
        df['activity_index'] = (
            df['projects_completed'] + 
            df['internships_completed'] + 
            df['hackathons_participated'] + 
            df['certifications_count']
        )
        
        # 4. Stress-Sleep Ratio: Wellness indicator
        df['stress_sleep_ratio'] = df['stress_level'] / (df['sleep_hours'] + 0.1)
        
        # 5. Engagement Score: Study commitment indicator
        df['engagement_score'] = (
            df['study_hours_per_day'] + 
            (df['attendance_percentage'] / 20)
        ) / 2
        
        # ============ ENCODING ============
        # Encode binary categorical variables
        le_gender = LabelEncoder()
        le_gender.fit(['Male', 'Female'])
        df['gender'] = le_gender.transform(df['gender'])
        
        le_part = LabelEncoder()
        le_part.fit(['No', 'Yes'])
        df['part_time_job'] = le_part.transform(df['part_time_job'])
        
        le_internet = LabelEncoder()
        le_internet.fit(['No', 'Yes'])
        df['internet_access'] = le_internet.transform(df['internet_access'])
        
        # One-hot encode multi-class categorical variables
        df = pd.get_dummies(
            df, 
            columns=['branch', 'city_tier', 'family_income_level', 'extracurricular_involvement'],
            drop_first=True,
            dtype=int
        )
        
        # Drop Student_ID and any non-numeric columns
        df = df.drop(columns=['Student_ID'], errors='ignore')
        
        # Ensure all columns are numeric
        numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        df = df[numeric_cols]
        
        return df
    
    def predict_placement(self, student_data: StudentData) -> dict:
        """Predict placement status."""
        if self.classification_model is None:
            raise HTTPException(status_code=503, detail="Classification model not available")
        
        df = self.preprocess_input(student_data)
        
        if self.classification_features:
            for col in self.classification_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.classification_features]
        
        prediction = self.classification_model.predict(df)[0]
        
        try:
            probability = self.classification_model.predict_proba(df)[0].max()
        except:
            probability = 0.5
        
        return {
            'student_id': student_data.Student_ID,
            'prediction': 'Placed' if prediction == 1 else 'Not Placed',
            'confidence': float(probability),
            'prediction_type': 'classification'
        }
    
    def predict_salary(self, student_data: StudentData) -> dict:
        """Predict salary."""
        if self.regression_model is None:
            raise HTTPException(status_code=503, detail="Regression model not available")
        
        df = self.preprocess_input(student_data)
        
        if self.regression_features:
            for col in self.regression_features:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.regression_features]
        
        prediction = self.regression_model.predict(df)[0]
        
        return {
            'student_id': student_data.Student_ID,
            'prediction': float(prediction),
            'confidence': 0.85,
            'prediction_type': 'regression'
        }


# Initialize model registry
model_registry = ModelRegistry()


# Routes
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "active",
        "service": "Student Placement Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "placement": "/predict/placement",
            "salary": "/predict/salary",
            "batch": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check with model status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "classification": "available" if model_registry.classification_model else "unavailable",
            "regression": "available" if model_registry.regression_model else "unavailable"
        }
    }


@app.post("/predict/placement", response_model=PredictionResponse, tags=["Predictions"])
async def predict_placement(student: StudentData):
    """
    Predict placement status for a student.
    
    **Test Case 1:** Pass required fields to get placement prediction
    
    Returns:
    - student_id: Echo of input student ID
    - prediction: "Placed" or "Not Placed"
    - confidence: Confidence score (0-1)
    - timestamp: Prediction timestamp
    """
    try:
        result = model_registry.predict_placement(student)
        
        return PredictionResponse(
            student_id=result['student_id'],
            prediction_type='placement',
            prediction=result['prediction'],
            confidence=result['confidence'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Placement prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/salary", response_model=PredictionResponse, tags=["Predictions"])
async def predict_salary(student: StudentData):
    """
    Predict salary for a student.
    
    **Test Case 2:** Pass required fields to get salary prediction
    
    Returns:
    - student_id: Echo of input student ID
    - prediction: Predicted salary in LPA
    - confidence: Confidence score (0-1)
    - timestamp: Prediction timestamp
    """
    try:
        result = model_registry.predict_salary(student)
        
        return PredictionResponse(
            student_id=result['student_id'],
            prediction_type='salary',
            prediction=result['prediction'],
            confidence=result['confidence'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Salary prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(
    file: UploadFile = File(...),
    prediction_type: str = "both"
):
    """
    Batch prediction from CSV file.
    
    **Test Case 3:** Upload CSV file with multiple student records
    
    Args:
    - file: CSV file with student data
    - prediction_type: "placement", "salary", or "both"
    
    Returns:
    - total_records: Total records in file
    - successful: Successfully predicted records
    - failed: Failed predictions
    - results: List of predictions
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        results = []
        successful = 0
        failed = 0
        
        for idx, row in df.iterrows():
            try:
                student = StudentData(**row.to_dict())
                
                if prediction_type in ['placement', 'both']:
                    placement = model_registry.predict_placement(student)
                    results.append(placement)
                    successful += 1
                
                if prediction_type in ['salary', 'both']:
                    salary = model_registry.predict_salary(student)
                    results.append(salary)
                    successful += 1
            
            except Exception as e:
                logger.error(f"Row {idx} failed: {str(e)}")
                failed += 1
        
        return BatchPredictionResponse(
            total_records=len(df),
            successful=successful,
            failed=failed,
            results=results if results else []
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stats", tags=["Statistics"])
async def get_stats():
    """Get API statistics and model information."""
    return {
        "api": {
            "title": "Student Placement & Salary Prediction API",
            "version": "1.0.0",
            "endpoints": 5
        },
        "models": {
            "classification": {
                "status": "available" if model_registry.classification_model else "unavailable",
                "type": "Classification",
                "target": "Placement Status"
            },
            "regression": {
                "status": "available" if model_registry.regression_model else "unavailable",
                "type": "Regression",
                "target": "Salary (LPA)"
            }
        },
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("🚀 STARTING FASTAPI SERVER")
    print("=" * 80)
    print("\n📊 API Documentation: http://localhost:8000/docs")
    print("📋 ReDoc Documentation: http://localhost:8000/redoc")
    print("⚙️  Health Check: http://localhost:8000/health")
    print("\n" + "=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
