"""
Streamlit Frontend

Client application communicating with FastAPI backend.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Student Placement System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"


class APIClient:
    """Client for communicating with FastAPI backend."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def health_check(self) -> dict:
        """Check API health status."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def predict_placement(self, student_data: dict) -> dict:
        """Get placement prediction from API."""
        try:
            response = requests.post(
                f"{self.base_url}/predict/placement",
                json=student_data,
                timeout=10
            )
            
            if response.status_code == 200:
                return {'success': True, 'data': response.json()}
            else:
                return {'success': False, 'error': response.text}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_salary(self, student_data: dict) -> dict:
        """Get salary prediction from API."""
        try:
            response = requests.post(
                f"{self.base_url}/predict/salary",
                json=student_data,
                timeout=10
            )
            
            if response.status_code == 200:
                return {'success': True, 'data': response.json()}
            else:
                return {'success': False, 'error': response.text}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def batch_predict(self, file, prediction_type: str = "both") -> dict:
        """Send batch prediction request."""
        try:
            files = {'file': file}
            params = {'prediction_type': prediction_type}
            
            response = requests.post(
                f"{self.base_url}/predict/batch",
                files=files,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return {'success': True, 'data': response.json()}
            else:
                return {'success': False, 'error': response.text}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Initialize API client
api_client = APIClient(API_BASE_URL)


def create_header():
    """Create application header with API status."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Student Placement System")
        st.markdown("FastAPI Backend + Streamlit Frontend")
    
    with col2:
        health = api_client.health_check()
        if health:
            st.success("API Connected")
            st.caption(f"Service: Active")
        else:
            st.error("API Disconnected")
            st.caption(f"Backend unavailable at {API_BASE_URL}")


def create_student_form() -> dict:
    """Create student data input form."""
    st.subheader("Student Information Form")
    
    col1, col2, col3 = st.columns(3)
    
    form_data = {}
    
    with col1:
        st.subheader("Basic Information")
        form_data['Student_ID'] = st.number_input("Student ID", min_value=1, value=1)
        form_data['gender'] = st.selectbox("Gender", ["Male", "Female"])
        form_data['branch'] = st.selectbox("Branch", ["CSE", "ECE", "IT", "CE"])
        form_data['city_tier'] = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        form_data['family_income_level'] = st.selectbox("Family Income", ["Low", "Medium", "High"])
    
    with col2:
        st.subheader("Academic Performance")
        form_data['cgpa'] = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        form_data['tenth_percentage'] = st.slider("10th Percentage", 0.0, 100.0, 75.0, 0.1)
        form_data['twelfth_percentage'] = st.slider("12th Percentage", 0.0, 100.0, 75.0, 0.1)
        form_data['backlogs'] = st.number_input("Backlogs", min_value=0, value=0)
        form_data['attendance_percentage'] = st.slider("Attendance %", 0.0, 100.0, 75.0, 1.0)
    
    with col3:
        st.subheader("Skills & Experience")
        form_data['coding_skill_rating'] = st.slider("Coding Skill (1-5)", 1, 5, 3)
        form_data['communication_skill_rating'] = st.slider("Communication (1-5)", 1, 5, 3)
        form_data['aptitude_skill_rating'] = st.slider("Aptitude (1-5)", 1, 5, 3)
        form_data['projects_completed'] = st.number_input("Projects", min_value=0, value=5)
        form_data['internships_completed'] = st.number_input("Internships", min_value=0, value=2)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("Engagement")
        form_data['study_hours_per_day'] = st.slider("Study Hours/Day", 0.0, 12.0, 4.0, 0.5)
        form_data['hackathons_participated'] = st.number_input("Hackathons", min_value=0, value=2)
        form_data['certifications_count'] = st.number_input("Certifications", min_value=0, value=2)
    
    with col5:
        st.subheader("Lifestyle")
        form_data['sleep_hours'] = st.slider("Sleep Hours/Day", 0.0, 12.0, 7.0, 0.5)
        form_data['stress_level'] = st.slider("Stress Level (1-10)", 1, 10, 5)
        form_data['part_time_job'] = st.selectbox("Part-time Job", ["No", "Yes"])
    
    with col6:
        st.subheader("Other")
        form_data['internet_access'] = st.selectbox("Internet Access", ["No", "Yes"])
        form_data['extracurricular_involvement'] = st.selectbox(
            "Extracurricular", ["Low", "Medium", "High"]
        )
    
    return form_data


def show_classification_prediction():
    """Show classification (placement) prediction."""
    st.subheader("CLASSIFICATION: Placement Prediction")
    
    st.info("""
    **Task:** Predict whether a student will be placed or not.
    
    **Test Scenario 1:** High-performing student with strong skills
    - Expected: Placed with high confidence
    """)
    
    form_data = create_student_form()
    
    if st.button("🔮 Predict Placement Status", use_container_width=True, key="clf_predict"):
        with st.spinner("Connecting to API and making prediction..."):
            result = api_client.predict_placement(form_data)
            
            if result['success']:
                pred = result['data']
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        pred['prediction'],
                        delta="Confident" if pred['confidence'] > 0.75 else "Moderate"
                    )
                
                with col2:
                    st.metric("Confidence Score", f"{pred['confidence']*100:.1f}%")
                
                with col3:
                    st.metric("Student ID", pred['student_id'])
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 4))
                categories = ['Confidence for Placement']
                confidence = [pred['confidence']]
                colors = ['#2ecc71' if pred['prediction'] == 'Placed' else '#e74c3c']
                bars = ax.barh(categories, confidence, color=colors, alpha=0.8)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Confidence Score')
                ax.set_title(f"Prediction: {pred['prediction']}")
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f'{width:.1%}', ha='left', va='center', fontweight='bold')
                
                st.pyplot(fig, use_container_width=True)
                
                st.success(f"Prediction completed at {pred['timestamp']}")
            
            else:
                st.error(f"Error: {result['error']}")


def show_regression_prediction():
    """Show regression (salary) prediction."""
    st.subheader("REGRESSION: Salary Estimation")
    
    st.info("""
    **Task:** Predict the expected salary for a student.
    
    **Test Scenario 2:** Mid-level student with moderate skills
    - Expected: Moderate salary prediction with reasonable confidence
    """)
    
    form_data = create_student_form()
    
    if st.button("Predict Salary", use_container_width=True, key="reg_predict"):
        with st.spinner("Connecting to API and making prediction..."):
            result = api_client.predict_salary(form_data)
            
            if result['success']:
                pred = result['data']
                
                salary = pred['prediction']
                salary_low = salary * 0.9
                salary_high = salary * 1.1
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Salary", f"₹{salary:.2f}L")
                
                with col2:
                    st.metric("Range (Low)", f"₹{salary_low:.2f}L")
                
                with col3:
                    st.metric("Range (High)", f"₹{salary_high:.2f}L")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(['Salary'], [salary], color='#3498db', alpha=0.8, label='Point Estimate')
                ax.barh(['Range'], [salary_high - salary_low], left=salary_low,
                       color='#95a5a6', alpha=0.4, label='±10% Range')
                ax.set_xlabel('Salary (LPA)')
                ax.legend()
                ax.set_title('Salary Prediction')
                
                st.pyplot(fig, use_container_width=True)
                
                st.success(f"✅ Prediction completed at {pred['timestamp']}")
            
            else:
                st.markdown(f'<div class="error-box">❌ Error: {result["error"]}</div>', 
                           unsafe_allow_html=True)


def show_batch_processing():
    """Show batch prediction processing."""
    st.markdown('<p class="section-header">📤 Batch Prediction</p>', unsafe_allow_html=True)
    
    st.info("Upload a CSV file with multiple student records for batch processing.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"**Preview:** {len(df_preview)} records")
        st.dataframe(df_preview.head(3))
        
        prediction_type = st.radio(
            "Select prediction type",
            ["Placement", "Salary", "Both"]
        )
        
        type_map = {
            "Placement": "placement",
            "Salary": "salary",
            "Both": "both"
        }
        
        if st.button("🔄 Process Batch", use_container_width=True):
            with st.spinner("Processing batch predictions..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                result = api_client.batch_predict(uploaded_file, type_map[prediction_type])
                
                if result['success']:
                    data = result['data']
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", data['total_records'])
                    with col2:
                        st.metric("Successful", data['successful'])
                    with col3:
                        st.metric("Failed", data['failed'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display results
                    if data['results']:
                        results_df = pd.DataFrame(data['results'])
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                
                else:
                    st.markdown(f'<div class="error-box">❌ Error: {result["error"]}</div>', 
                               unsafe_allow_html=True)


def show_api_documentation():
    """Show API documentation and endpoints."""
    st.markdown('<p class="section-header">📚 API Documentation</p>', unsafe_allow_html=True)
    
    st.info(f"**API Base URL:** `{API_BASE_URL}`")
    
    tabs = st.tabs(["Endpoints", "Request/Response", "Test Cases"])
    
    with tabs[0]:
        st.markdown("""
        #### Available Endpoints
        
        1. **GET /health**
           - Health check endpoint
           - Returns: API status and model availability
        
        2. **POST /predict/placement**
           - Predict placement status
           - Input: StudentData
           - Output: PredictionResponse
        
        3. **POST /predict/salary**
           - Predict salary
           - Input: StudentData
           - Output: PredictionResponse
        
        4. **POST /predict/batch**
           - Batch prediction from CSV
           - Input: CSV file
           - Output: BatchPredictionResponse
        
        5. **GET /api/stats**
           - Get API statistics
           - Returns: Model info and stats
        """)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sample Request (Placement)**")
            st.json({
                "Student_ID": 1,
                "gender": "Male",
                "branch": "CSE",
                "cgpa": 7.5,
                "tenth_percentage": 75.0,
                "twelfth_percentage": 75.0,
                "backlogs": 0,
                "study_hours_per_day": 4.0,
                "attendance_percentage": 75.0,
                "projects_completed": 5,
                "internships_completed": 2
            })
        
        with col2:
            st.markdown("**Sample Response**")
            st.json({
                "student_id": 1,
                "prediction_type": "placement",
                "prediction": "Placed",
                "confidence": 0.85,
                "timestamp": "2024-04-21T10:30:00"
            })
    
    with tabs[2]:
        st.markdown("""
        #### Test Cases (3 Minimum Required)
        
        **Test Case 1: Classification (Placement)**
        - ✅ Implemented in "Classification" tab
        - Method: POST /predict/placement
        - Scenario: Single student prediction
        
        **Test Case 2: Regression (Salary)**
        - ✅ Implemented in "Regression" tab
        - Method: POST /predict/salary
        - Scenario: Single student prediction
        
        **Test Case 3: Batch Processing**
        - ✅ Implemented in "Batch" tab
        - Method: POST /predict/batch
        - Scenario: Multiple student predictions
        """)


def main():
    """Main application."""
    create_header()
    
    st.markdown("---")
    
    # Main navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        screen = st.radio(
            "Select Option",
            [
                "Classification (Placement)",
                "Regression (Salary)",
                "Batch Processing",
                "API Documentation"
            ]
        )
    
    if screen == "Classification (Placement)":
        show_classification_prediction()
    
    elif screen == "Regression (Salary)":
        show_regression_prediction()
    
    elif screen == "Batch Processing":
        show_batch_processing()
    
    else:
        show_api_documentation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d;">
        <p>🚀 Decoupled Architecture | FastAPI Backend + Streamlit Frontend</p>
        <p>Run: <code>streamlit run 05_streamlit_frontend.py</code></p>
        <p>Backend: <code>python 04_fastapi_backend.py</code></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
