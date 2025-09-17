# --------------------- app.py ---------------------
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# ---------------- Load Data ----------------
DATA_PATH = "students_dataset.csv"  # your CSV file
MODEL_PATH = "student_model.pkl"     # trained model
ENCODER_PATH = "label_encoders.pkl"  # label encoders

# Load dataset
df = pd.read_csv(DATA_PATH)

# Load model & encoders
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# ---------------- Prediction Function ----------------
def predict_student(student_data):
    df_input = pd.DataFrame([student_data])
    for col, le in label_encoders.items():
        if col in df_input.columns:
            df_input[col] = le.transform(df_input[col])
    prediction = model.predict(df_input)[0]
    return prediction

# ---------------- Streamlit UI ----------------
st.title("🎓 Student Performance Prediction & Dashboard")

tab1, tab2 = st.tabs(["Predict", "Dashboard"])

# ---------------- Prediction Tab ----------------
with tab1:
    st.header("Predict Student Assessment Score")
    class_choice = st.selectbox("Class", df["class"].unique())
    comprehension = st.slider("Comprehension", 0, 100, 70)
    attention = st.slider("Attention", 0, 100, 70)
    focus = st.slider("Focus", 0, 100, 70)
    retention = st.slider("Retention", 0, 100, 70)
    engagement_time = st.slider("Engagement Time (mins)", 0, 180, 90)

    student_data = {
        "class": class_choice,
        "comprehension": comprehension,
        "attention": attention,
        "focus": focus,
        "retention": retention,
        "engagement_time": engagement_time
    }

    if st.button("Predict"):
        result = predict_student(student_data)
        st.success(f"Predicted Assessment Score: **{result}**")

# ---------------- Dashboard Tab ----------------
with tab2:
    st.header("📊 Student Dashboard")

    # Overview stats
    st.subheader("Overview Stats")
    st.dataframe(df.describe())

    # Bar chart for average skills
    st.subheader("Average Skill Scores")
    avg_skills = df[["comprehension", "attention", "focus", "retention", "engagement_time"]].mean()
    st.bar_chart(avg_skills)

    # Scatter plot: Attention vs Assessment Score
    st.subheader("Attention vs Assessment Score")
    fig, ax = plt.subplots()
    sns.scatterplot(x="attention", y="assessment_score", data=df, ax=ax)
    st.pyplot(fig)

    # Radar chart for first student
    st.subheader("Sample Student Profile (Radar Chart)")
    student = df.iloc[0]
    fig = px.line_polar(
        r=[student.comprehension, student.attention, student.focus, student.retention, student.engagement_time],
        theta=["Comprehension", "Attention", "Focus", "Retention", "Engagement Time"],
        line_close=True
    )
    st.plotly_chart(fig)

    # Clustering into Learning Personas
    st.subheader("Learning Personas (Clusters)")
    skills = ["comprehension", "attention", "focus", "retention", "engagement_time"]
    X_scaled = StandardScaler().fit_transform(df[skills])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["persona"] = kmeans.fit_predict(X_scaled)
    st.bar_chart(df["persona"].value_counts())
