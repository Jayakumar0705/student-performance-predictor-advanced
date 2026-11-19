
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

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
# 3. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    data = pd.read_csv("student_data.csv")

    required_columns = ["Hours", "Score"]

    if not all(column in data.columns for column in required_columns):
        raise ValueError(
            "Dataset must contain Hours and Score columns."
        )

    data = data[required_columns].copy()

    data["Hours"] = pd.to_numeric(
        data["Hours"], errors="coerce"
    )

    data["Score"] = pd.to_numeric(
        data["Score"], errors="coerce"
    )

    data = data.dropna()

    if len(data) < 3:
        raise ValueError("At least 3 valid student records are required.")

    return data


df = load_data()


# ==========================================
# 4. TRAIN MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def train_model(data):

    X = data[["Hours"]]
    y = data["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    # Predictions on test records
    y_pred = model.predict(X_test)

    # Evaluation metrics
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
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🎓 Student Predictor")

st.sidebar.caption("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Score Prediction",
        "Performance Analysis",
        "Dataset & Model"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Predict student exam performance "
    "using Machine Learning."
)


# ==========================================
# 6. DASHBOARD
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

    line_scores = model.predict(
        pd.DataFrame({"Hours": line_hours})
    )

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
# 7. SCORE PREDICTION
# ==========================================

elif page == "Score Prediction":

    st.title("Student Score Prediction")

    st.write(
        "Enter the number of hours you study "
        "to estimate your exam score."
    )

    hours = st.slider(
        "Select Study Hours",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )

    if st.button("Predict Exam Score", type="primary"):

        prediction = model.predict(
            pd.DataFrame({
                "Hours": [hours]
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
        df[["Hours"]]
    )

    # Create comparison dataset

    results = pd.DataFrame({

        "Study Hours": df["Hours"].values,

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
