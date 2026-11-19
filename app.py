
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

# ==========================================
# PDF REPORT GENERATOR
# ==========================================

def generate_student_pdf(
    student_name,
    hours,
    attendance,
    assignment,
    previous_score,
    prediction
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "STUDENT PERFORMANCE REPORT",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            f"<b>Student Name:</b> {student_name}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    table_data = [
        ["Student Parameter", "Value"],
        ["Study Hours", f"{hours} hours"],
        ["Attendance", f"{attendance}%"],
        ["Assignment Marks", f"{assignment}/100"],
        ["Previous Exam Score", f"{previous_score}/100"],
        ["Predicted Exam Score", f"{prediction:.2f}/100"]
    ]

    table = Table(
        table_data,
        colWidths=[250, 200]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ])
    )

    story.append(table)

    story.append(Spacer(1, 25))

    if prediction >= 80:
        performance = "Excellent"
    elif prediction >= 60:
        performance = "Good"
    elif prediction >= 40:
        performance = "Average"
    else:
        performance = "Needs Improvement"

    story.append(
        Paragraph(
            f"<b>Performance Category:</b> {performance}",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "This report contains an estimated exam score "
            "generated using a Multi-Feature Linear Regression model.",
            styles["Normal"]
        )
    )

    document.build(story)

    return buffer.getvalue()

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# CUSTOM CSS - MODERN WHITE AND BLUE DESIGN
# ==========================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #F8FAFC;
}

/* Main content spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Headings */

h1 {
    color: #172554;
    font-weight: 750;
    letter-spacing: -1px;
}

h2, h3 {
    color: #1E3A8A;
    font-weight: 650;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

section[data-testid="stSidebar"] h1 {
    color: #1E40AF;
    font-size: 25px;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 3px 12px rgba(30, 64, 175, 0.05);
}

div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 500;
}

div[data-testid="stMetricValue"] {
    color: #1D4ED8;
    font-weight: 750;
}

/* Buttons */

.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 22px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
    border: none;
}

/* Input widgets */

.stSlider {
    padding-top: 12px;
    padding-bottom: 12px;
}

/* Dataframes */

div[data-testid="stDataFrame"] {
    border: 1px solid #DBEAFE;
    border-radius: 12px;
    overflow: hidden;
}

/* Graph containers */

div[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 12px;
}

/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dividers */

hr {
    border-color: #E2E8F0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

st.sidebar.subheader("Upload Student Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

@st.cache_data
def load_data(file):

    data = pd.read_csv(file)

    required_columns = [
        "Hours",
        "Attendance",
        "Assignment",
        "Previous_Score",
        "Score"
    ]

    if not all(column in data.columns for column in required_columns):
        raise ValueError(
            "CSV must contain: Hours, Attendance, "
            "Assignment, Previous_Score, Score."
        )

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna()

    if len(data) < 5:
        raise ValueError(
            "At least 5 valid student records are required."
        )

    return data


try:

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")

    else:
        df = load_data("advanced_student_data.csv")

except Exception as e:

    st.error(f"Dataset Error: {e}")
    st.stop()


# ==========================================
# 3. TRAIN MACHINE LEARNING MODEL
# ==========================================

feature_columns = [
    "Hours",
    "Attendance",
    "Assignment",
    "Previous_Score"
]

@st.cache_resource
def train_model(data):

    X = data[feature_columns]
    y = data["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return (
        model,
        mae,
        mse,
        rmse,
        r2,
        X_test,
        y_test,
        y_pred
    )


(
    model,
    mae,
    mse,
    rmse,
    r2,
    X_test,
    y_test,
    y_pred
) = train_model(df)

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🎓 Student Predictor")

st.sidebar.caption("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Data Entry",
        "Score Prediction",
        "Performance Analysis",
        "Dataset & Model",
        "Model Comparison"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Predict student exam performance "
    "using Machine Learning."
)


# ==========================================
# 5. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("Student Performance Dashboard")

    st.write(
        "Welcome! Analyze student performance "
        "and explore exam score predictions."
    )

    # Dashboard metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Students",
        len(df)
    )

    col2.metric(
        "Average Score",
        f"{df['Score'].mean():.2f}"
    )

    col3.metric(
        "Average Study Hours",
        f"{df['Hours'].mean():.2f}"
    )

    col4.metric(
        "Highest Score",
        f"{df['Score'].max():.0f}"
    )

    st.divider()

    # Study hours graph

    st.subheader("Study Hours vs Exam Scores")

    fig = px.scatter(
        df,
        x="Hours",
        y="Score",
        title="Relationship Between Study Hours and Exam Scores",
        labels={
            "Hours": "Study Hours",
            "Score": "Exam Score"
        },
        template="plotly_white",
        color_discrete_sequence=["#6366F1"]
    )

    # Add regression line

    line_hours = np.linspace(
        df["Hours"].min(),
        df["Hours"].max(),
        100
    )

    line_data = pd.DataFrame({
        "Hours": line_hours,
        "Attendance": df["Attendance"].mean(),
        "Assignment": df["Assignment"].mean(),
        "Previous_Score": df["Previous_Score"].mean()
    })

    line_scores = model.predict(line_data)

    fig.add_scatter(
        x=line_hours,
        y=line_scores,
        mode="lines",
        name="Regression Line",
        line=dict(
            color="#2563EB",
            width=3
        )
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, width="stretch")

    # Dataset overview

    st.subheader("Student Dataset Overview")

    st.dataframe(
        df,
        width="stretch"
    )



# ==========================================
# 6. INDIVIDUAL STUDENT DATA ENTRY
# ==========================================

elif page == "Student Data Entry":

    st.title("Individual Student Data Entry")

    st.write(
        "Enter student details to estimate exam performance."
    )

    with st.form("student_entry_form"):

        student_name = st.text_input("Student Name")

        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )

        attendance = st.slider(
            "Attendance Percentage",
            min_value=0,
            max_value=100,
            value=75
        )

        assignment_marks = st.number_input(
            "Assignment Marks",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

        previous_score = st.number_input(
            "Previous Exam Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        submitted = st.form_submit_button(
            "Predict Student Performance"
        )

    if submitted:

        if not student_name.strip():
            st.error("Please enter the student's name.")

        else:

            
            prediction = model.predict(
                pd.DataFrame({
                    "Hours": [study_hours],
                    "Attendance": [attendance],
                    "Assignment": [assignment_marks],
                    "Previous_Score": [previous_score]
                })
            )[0]

            prediction = max(0, min(100, prediction))

            st.divider()

            st.subheader("Student Prediction Report")

            st.write(f"**Student Name:** {student_name}")

            st.metric(
                "Predicted Exam Score",
                f"{prediction:.2f}/100"
            )

            st.write(f"**Study Hours:** {study_hours}")

            st.write(f"**Attendance:** {attendance}%")

            st.write(f"**Assignment Marks:** {assignment_marks}/100")

            st.write(f"**Previous Exam Score:** {previous_score}/100")

            st.info(
                "The prediction uses study hours, attendance, assignment marks, "
                "and previous exam score."
            )


            # Generate PDF report

            pdf_report = generate_student_pdf(
                student_name,
                study_hours,
                attendance,
                assignment_marks,
                previous_score,
                prediction
            )

            st.download_button(
                label="Download Student Report (PDF)",
                data=pdf_report,
                file_name=f"{student_name}_Performance_Report.pdf",
                mime="application/pdf",
                type="primary"
            )



# ==========================================
# 7. SCORE PREDICTION
# ==========================================

elif page == "Score Prediction":

    st.title("Student Score Prediction")

    st.write(
        "Enter the number of hours you study "
        "to estimate your exam score."
    )

    
    hours = st.slider(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance Percentage",
        min_value=0,
        max_value=100,
        value=75
    )

    assignment = st.number_input(
        "Assignment Marks",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    previous_score = st.number_input(
        "Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )


    if st.button("Predict Exam Score", type="primary"):

        
        prediction = model.predict(
            pd.DataFrame({
                "Hours": [hours],
                "Attendance": [attendance],
                "Assignment": [assignment],
                "Previous_Score": [previous_score]
            })
        )[0]


        prediction = max(
            0,
            min(100, prediction)
        )

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Exam Score",
            f"{prediction:.2f}/100"
        )

        col2.metric(
            "Study Duration",
            f"{hours:.1f} Hours"
        )

        # Performance classification

        if prediction >= 80:

            performance = "Excellent"

            advice = (
                "Maintain your study routine "
                "and continue practising."
            )

        elif prediction >= 60:

            performance = "Good"

            advice = (
                "Keep improving through regular "
                "revision and practice."
            )

        elif prediction >= 40:

            performance = "Average"

            advice = (
                "Increase focused study time "
                "and practise more questions."
            )

        else:

            performance = "Needs Improvement"

            advice = (
                "Build a consistent study schedule "
                "and seek guidance."
            )

        st.success(
            f"Predicted Performance: {performance}"
        )

        st.info(
            f"Recommendation: {advice}"
        )

        st.caption(
            "This is an estimated score generated by "
            "a Linear Regression model trained on "
            "the available sample dataset."
        )


# ==========================================
# 8. PERFORMANCE ANALYSIS
# ==========================================

elif page == "Performance Analysis":

    st.title("Performance Analysis")

    st.write(
        "Analyze the Machine Learning model's performance."
    )

    # Display test-set evaluation metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    col2.metric(
        "MSE",
        f"{mse:.2f}"
    )

    col3.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    col4.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    st.caption(
        "These metrics are calculated using the "
        "independent test split."
    )

    st.divider()

    # Generate predictions for all student records

    
    all_predictions = model.predict(
        df[feature_columns]
    )


    # Create comparison dataset

    
    results = pd.DataFrame({

    "Study Hours": df["Hours"].values,

    "Attendance": df["Attendance"].values,

    "Assignment Marks": df["Assignment"].values,

    "Previous Exam Score": df["Previous_Score"].values,

    "Actual Score": df["Score"].values,

    "Predicted Score": all_predictions

})


    # Calculate prediction error

    results["Prediction Error"] = (
        results["Actual Score"]
        - results["Predicted Score"]
    ).abs()

    # Display graph

    st.subheader(
        "Actual vs Predicted Scores - All Students"
    )

    fig = px.scatter(
        results,
        x="Actual Score",
        y="Predicted Score",
        hover_data=[
            "Study Hours",
            "Prediction Error"
        ],
        title="Actual vs Model-Fitted Exam Scores",
        template="plotly_white",
        color_discrete_sequence=["#2563EB"]
    )

    # Perfect prediction reference line

    min_score = min(
        results["Actual Score"].min(),
        results["Predicted Score"].min()
    )

    max_score = max(
        results["Actual Score"].max(),
        results["Predicted Score"].max()
    )

    fig.add_shape(
        type="line",
        x0=min_score,
        y0=min_score,
        x1=max_score,
        y1=max_score,
        line=dict(
            color="red",
            dash="dash",
            width=2
        )
    )

    fig.update_layout(
        height=500,
        xaxis_title="Actual Exam Score",
        yaxis_title="Model-Fitted Exam Score"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # Display prediction table

    st.subheader(
        "All Student Prediction Records"
    )

    st.dataframe(
        results.round(2),
        width="stretch"
    )


    # Download prediction records as CSV

    csv_data = results.round(2).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download All Student Records (CSV)",
        data=csv_data,
        file_name="student_performance_report.csv",
        mime="text/csv",
        type="primary"
    )


    st.caption(
        "The graph displays fitted predictions for all "
        "student records. The evaluation metrics above "
        "are calculated separately using only the test records. "
        "Training-set fitted predictions are not independent "
        "evidence of model performance."
    )


# ==========================================
# 9. DATASET AND MODEL INFORMATION
# ==========================================

elif page == "Dataset & Model":

    st.title(
        "Dataset & Model Information"
    )

    # Student dataset

    st.subheader("Student Dataset")

    st.dataframe(
        df,
        width="stretch"
    )

    # Statistical summary

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        width="stretch"
    )

    # Model information

    st.subheader(
        "Machine Learning Algorithm"
    )

    st.write(
        "Algorithm: Linear Regression"
    )

    st.write(
        f"Model Slope: {model.coef_[0]:.4f}"
    )

    st.write(
        f"Model Intercept: {model.intercept_:.4f}"
    )

    # Prediction equation

    st.subheader(
        "Prediction Equation"
    )

    st.code(
        f"Score = {model.coef_[0]:.4f} × Hours "
        f"+ {model.intercept_:.4f}"
    )

    st.success(
        "Machine Learning model trained successfully!"
    )

    st.info(
        f"Total records used: {len(df)}"
    )


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":

    st.title("Machine Learning Model Comparison")

    st.write(
        "Compare Linear Regression, Decision Tree "
        "and Random Forest algorithms."
    )

    # Use the same train-test split for every model

    X = df[feature_columns]
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Initialize algorithms

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=3,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
    }

    comparison_results = []

    # Train and evaluate each model

    for model_name, comparison_model in models.items():

        comparison_model.fit(X_train, y_train)

        predictions = comparison_model.predict(X_test)

        model_mae = mean_absolute_error(
            y_test, predictions
        )

        model_mse = mean_squared_error(
            y_test, predictions
        )

        model_rmse = np.sqrt(model_mse)

        model_r2 = r2_score(
            y_test, predictions
        )

        comparison_results.append({
            "Algorithm": model_name,
            "MAE": round(model_mae, 2),
            "RMSE": round(model_rmse, 2),
            "R2 Score": round(model_r2, 3)
        })

    # Create comparison table

    comparison_df = pd.DataFrame(comparison_results)

    st.subheader("Model Performance Results")

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True
    )

    # MAE comparison chart

    st.subheader("MAE Comparison")

    mae_chart = px.bar(
        comparison_df,
        x="Algorithm",
        y="MAE",
        title="Mean Absolute Error",
        template="plotly_white",
        color="Algorithm"
    )

    st.plotly_chart(
        mae_chart,
        width="stretch"
    )

    # RMSE comparison chart

    st.subheader("RMSE Comparison")

    rmse_chart = px.bar(
        comparison_df,
        x="Algorithm",
        y="RMSE",
        title="Root Mean Squared Error",
        template="plotly_white",
        color="Algorithm"
    )

    st.plotly_chart(
        rmse_chart,
        width="stretch"
    )

    # R2 comparison chart

    st.subheader("R2 Score Comparison")

    r2_chart = px.bar(
        comparison_df,
        x="Algorithm",
        y="R2 Score",
        title="R2 Score",
        template="plotly_white",
        color="Algorithm"
    )

    st.plotly_chart(
        r2_chart,
        width="stretch"
    )

    st.info(
        "Lower MAE and RMSE indicate smaller prediction errors. "
        "Higher R2 indicates a better fit. Results are illustrative "
        "because the sample dataset is very small."
    )

