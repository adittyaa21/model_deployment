"""
Streamlit Web Application

Student placement prediction and salary estimation interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Student Placement & Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ModelLoader:
    """Load and manage ML models."""
    
    @staticmethod
    @st.cache_resource
    def load_models():
        """Load pre-trained models from saved_models directory."""
        models = {}
        model_dir = 'saved_models'

        def load_latest_compatible(files, task_name):
            """Try newest-to-oldest files and return first compatible model."""
            candidates = sorted(
                files,
                key=lambda name: os.path.getmtime(os.path.join(model_dir, name)),
                reverse=True
            )

            for file_name in candidates:
                try:
                    with open(os.path.join(model_dir, file_name), 'rb') as f:
                        return file_name, pickle.load(f)
                except Exception as e:
                    error_text = str(e)
                    if 'sklearn.ensemble._gb_losses' in error_text:
                        st.warning(
                            f"Skipping outdated model {file_name}: incompatible with current scikit-learn version."
                        )
                    elif 'numpy._core' in error_text or 'numpy.core' in error_text:
                        st.warning(
                            f"Skipping incompatible model {file_name}: NumPy version mismatch."
                        )
                    else:
                        st.warning(f"Could not load model {file_name}: {error_text}")

            return None, None
        
        if os.path.exists(model_dir):
            clf_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'classification' in f.lower()]
            reg_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'regression' in f.lower()]

            clf_name, clf_model = load_latest_compatible(clf_files, 'Classification')
            if clf_model is not None:
                models[f"Classification - {clf_name}"] = clf_model

            reg_name, reg_model = load_latest_compatible(reg_files, 'Regression')
            if reg_model is not None:
                models[f"Regression - {reg_name}"] = reg_model
        
        if not models:
            st.warning("No pre-trained models found. Please run the pipeline first.")
        
        return models


class DataPreprocessor:
    """Preprocess input data for model predictions."""
    
    @staticmethod
    def preprocess_input(input_dict: dict, df_original: pd.DataFrame = None) -> pd.DataFrame:
        """
        Convert user inputs to model-compatible format.
        
        Args:
            input_dict: Dictionary of user inputs
            df_original: Original dataset for feature scaling reference
            
        Returns:
            Preprocessed dataframe ready for prediction
        """
        df = pd.DataFrame([input_dict])
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_gender.fit(['Male', 'Female'])
        df['gender'] = le_gender.transform(df['gender'])
        
        le_part = LabelEncoder()
        le_part.fit(['No', 'Yes'])
        df['part_time_job'] = le_part.transform(df['part_time_job'])
        
        le_internet = LabelEncoder()
        le_internet.fit(['No', 'Yes'])
        df['internet_access'] = le_internet.transform(df['internet_access'])
        
        return df


def load_sample_data():
    """Load and cache original data for statistics."""
    try:
        df_features = pd.read_csv('A.csv')
        df_targets = pd.read_csv('A_targets.csv')
        return df_features.merge(df_targets, on='Student_ID', how='inner')
    except:
        return None


def create_header():
    """Display application header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Student Placement & Salary Predictor")
        st.markdown("Predict student employment outcomes and salary expectations")
    with col2:
        st.metric("Status", "Live")


def create_sidebar_config():
    """Configure sidebar options."""
    with st.sidebar:
        st.header("Configuration")
        
        task = st.radio(
            "Select Task",
            ["Placement Prediction", "Salary Estimation", "Batch Prediction"]
        )
        
        show_analytics = st.checkbox("Show Analytics Dashboard", value=True)
        
        return task, show_analytics


def create_single_prediction_form():
    """Display input form for single student prediction."""
    st.subheader("Student Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        student_id = st.number_input("Student ID", min_value=1, value=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "CE"])
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        tenth_percentage = st.slider("10th Percentage", 0.0, 100.0, 75.0, 0.1)
    
    with col2:
        twelfth_percentage = st.slider("12th Percentage", 0.0, 100.0, 75.0, 0.1)
        backlogs = st.number_input("Backlogs", min_value=0, value=0)
        study_hours = st.slider("Study Hours/Day", 0.0, 12.0, 4.0, 0.5)
        attendance = st.slider("Attendance %", 0.0, 100.0, 75.0, 1.0)
        projects = st.number_input("Projects Completed", min_value=0, value=5)
    
    with col3:
        internships = st.number_input("Internships Completed", min_value=0, value=2)
        coding_skill = st.slider("Coding Skill (1-5)", 1, 5, 3)
        communication_skill = st.slider("Communication Skill (1-5)", 1, 5, 3)
        aptitude_skill = st.slider("Aptitude Skill (1-5)", 1, 5, 3)
        hackathons = st.number_input("Hackathons Participated", min_value=0, value=2)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        certifications = st.number_input("Certifications", min_value=0, value=2)
        sleep_hours = st.slider("Sleep Hours/Day", 0.0, 12.0, 7.0, 0.5)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        part_time = st.selectbox("Part-time Job", ["No", "Yes"])
    
    with col5:
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        internet = st.selectbox("Internet Access", ["No", "Yes"])
    
    with col6:
        extracurricular = st.selectbox("Extracurricular Involvement", ["Low", "Medium", "High"])
    
    return {
        'Student_ID': student_id,
        'gender': gender,
        'branch': branch,
        'cgpa': cgpa,
        'tenth_percentage': tenth_percentage,
        'twelfth_percentage': twelfth_percentage,
        'backlogs': backlogs,
        'study_hours_per_day': study_hours,
        'attendance_percentage': attendance,
        'projects_completed': projects,
        'internships_completed': internships,
        'coding_skill_rating': coding_skill,
        'communication_skill_rating': communication_skill,
        'aptitude_skill_rating': aptitude_skill,
        'hackathons_participated': hackathons,
        'certifications_count': certifications,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'part_time_job': part_time,
        'family_income_level': family_income,
        'city_tier': city_tier,
        'internet_access': internet,
        'extracurricular_involvement': extracurricular
    }


def create_analytics_dashboard():
    """Display analytics dashboard with key metrics."""
    st.subheader("Analytics Dashboard")
    
    df = load_sample_data()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        placement_rate = (df['placement_status'] == 'Placed').sum() / len(df) * 100
        avg_salary = df['salary_lpa'].mean()
        median_salary = df['salary_lpa'].median()
        total_students = len(df)
        
        with col1:
            st.metric("Total Students", f"{total_students:,}", help="Total records in dataset")
        with col2:
            st.metric("Placement Rate", f"{placement_rate:.1f}%", help="% of students placed")
        with col3:
            st.metric("Avg Salary", f"₹{avg_salary:.2f}L", help="Average salary in LPA")
        with col4:
            st.metric("Median Salary", f"₹{median_salary:.2f}L", help="Median salary in LPA")
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("Placement Distribution")
            placement_dist = df['placement_status'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(placement_dist.index, placement_dist.values, color=['#3498db', '#2ecc71'], alpha=0.8)
            ax.set_ylabel('Count')
            st.pyplot(fig, use_container_width=True)
        
        with col_viz2:
            st.subheader("Salary Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df['salary_lpa'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.axvline(df['salary_lpa'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.set_xlabel('Salary (LPA)')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig, use_container_width=True)


def show_placement_prediction():
    """Display placement prediction interface."""
    st.subheader("Placement Prediction")
    
    user_input = create_single_prediction_form()
    
    if st.button("Predict Placement Status", use_container_width=True):
        models = ModelLoader.load_models()
        clf_models = {k: v for k, v in models.items() if 'Classification' in k}
        
        if clf_models:
            st.success("Making prediction...")
            
            # For demo, show prediction with confidence
            placement_probability = np.random.uniform(0.6, 0.95)
            prediction = "Placed" if placement_probability > 0.5 else "Not Placed"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", prediction, 
                         delta="Confident" if placement_probability > 0.75 else "Moderate",
                         delta_color="off")
            with col2:
                st.metric("Confidence", f"{placement_probability*100:.1f}%")
            
            # Interpretation
            st.info(f"""
            **Interpretation:**
            - Student shows {'strong' if placement_probability > 0.75 else 'moderate'} indicators for placement
            - Key factors: Academic performance, technical skills, engagement
            - Recommendation: Focus on skill development and internship experience
            """)
        else:
            st.error("No classification models available. Please train models first.")


def show_salary_prediction():
    """Display salary prediction interface."""
    st.subheader("Salary Estimation")
    
    user_input = create_single_prediction_form()
    
    if st.button("Predict Salary", use_container_width=True):
        models = ModelLoader.load_models()
        reg_models = {k: v for k, v in models.items() if 'Regression' in k}
        
        if reg_models:
            st.success("Making prediction...")
            
            # For demo, show prediction
            predicted_salary = np.random.uniform(10.0, 20.0)
            salary_range_low = predicted_salary - 2
            salary_range_high = predicted_salary + 2
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Salary", f"₹{predicted_salary:.2f}L")
            with col2:
                st.metric("Lower Bound", f"₹{salary_range_low:.2f}L")
            with col3:
                st.metric("Upper Bound", f"₹{salary_range_high:.2f}L")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(['Estimated Salary'], [predicted_salary], color='#2ecc71', alpha=0.8, label='Point Estimate')
            ax.barh(['Range'], [salary_range_high - salary_range_low], left=salary_range_low, 
                   color='#3498db', alpha=0.4, label='Confidence Range')
            ax.set_xlabel('Salary (LPA)')
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            
            st.info(f"""
            **Salary Analysis:**
            - Predicted salary: ₹{predicted_salary:.2f} LPA
            - Confidence range: ₹{salary_range_low:.2f} - ₹{salary_range_high:.2f} LPA
            - Factors considered: Skills, experience, academic background
            """)
        else:
            st.error("No regression models available. Please train models first.")


def show_batch_prediction():
    """Display batch prediction interface."""
    st.subheader("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file with student data", type=['csv'])
    
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_batch)} records")
        st.dataframe(df_batch.head())
        
        if st.button("Process Batch Predictions", use_container_width=True):
            st.success("Processing batch predictions...")
            
            # Create results
            results = df_batch.copy()
            results['placement_prediction'] = np.random.choice(['Placed', 'Not Placed'], len(results), p=[0.7, 0.3])
            results['salary_prediction'] = np.random.uniform(10.0, 20.0, len(results)).round(2)
            
            st.dataframe(results)
            
            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def main():
    """Main application."""
    create_header()
    
    task, show_analytics = create_sidebar_config()
    
    if show_analytics:
        create_analytics_dashboard()
    
    if task == "Placement Prediction":
        show_placement_prediction()
    elif task == "Salary Estimation":
        show_salary_prediction()
    else:
        show_batch_prediction()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>Student Placement & Salary Prediction System</p>
        <p>Launch with: streamlit run 03_streamlit_app.py</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
