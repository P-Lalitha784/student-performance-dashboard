# Student Performance Dashboard

This project analyzes students' cognitive skills and predicts their assessment score. It includes a Streamlit app for interactive predictions.

## Dataset
- Synthetic dataset with 100 students
- Columns: student_id, name, class, comprehension, attention, focus, retention, engagement_time, assessment_score

## Features
- Predict assessment score category (Pass/Fail/Excellent)
- Visualizations: correlation heatmap, feature importance, skill vs score charts
- Interactive Streamlit dashboard

## Setup
1. Clone the repository:
   ```bash
   git clone <YOUR_REPO_LINK>
   cd student-performance-dashboard
2.Install dependencies:

pip install -r requirements.txt

3.Run Streamlit app:

streamlit run app.py

Files:-

1.app.py - Streamlit app

2.notebook.ipynb - Jupyter notebook with EDA and ML

3.students_dataset.csv - Dataset

4.student_model.pkl - Trained ML model

5.label_encoders.pkl - Encoders for categorical features
