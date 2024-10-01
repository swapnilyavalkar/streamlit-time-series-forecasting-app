import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# App title
st.title("ABC Data Analysis App")

# Sidebar for data and model parameters
st.sidebar.header("1. Data")

# Upload dataset section
st.sidebar.subheader("Upload your time series data")
uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, or TXT file", type=['csv', 'xlsx', 'txt'])

if uploaded_file:
    # Reading the file based on its extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, delimiter='\t')

    # Sidebar options for selecting columns and setting data format
    st.sidebar.subheader("Columns")
    date_column = st.sidebar.selectbox("Select Date Column", options=[None] + list(df.columns), index=0)
    target_column = st.sidebar.selectbox("Select Target Column (Y)", options=[None] + list(df.columns), index=0)

    # Check if valid columns are selected
    if date_column is None or target_column is None:
        st.error("Please select both Date Column and Target Column.")
    else:
        # Rename columns for Prophet compatibility
        df = df.rename(columns={date_column: 'ds', target_column: 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Sidebar for additional options (placeholders)
        st.sidebar.subheader("Filtering")
        st.sidebar.subheader("Resampling")
        st.sidebar.subheader("Cleaning")

        # Modelling options in the sidebar
        st.sidebar.header("2. Modelling")
        st.sidebar.selectbox("Seasonalities", ["additive", "multiplicative"], index=0)
        st.sidebar.selectbox("Prior scale", ["auto", 10, 20], index=0)
        st.sidebar.subheader("Other parameters")

        # Evaluation settings in the sidebar
        st.sidebar.header("3. Evaluation")
        st.sidebar.subheader("Cross-validation split")
        training_start = st.sidebar.date_input("Training start date")
        training_end = st.sidebar.date_input("Training end date")
        validation_start = st.sidebar.date_input("Validation start date")
        validation_end = st.sidebar.date_input("Validation end date")

        st.sidebar.subheader("Metrics")
        st.sidebar.selectbox("Metrics to evaluate", ["RMSE", "MAE", "SMAPE"], index=0)

        # Forecast generation button
        launch_forecast = st.sidebar.checkbox("Launch forecast")
        track_experiments = st.sidebar.checkbox("Track experiments")

        # Main content section - Overview of the data and plots
        st.header("1. Overview")

        st.write("""
            **Visualization information:**
            - The blue line shows the predictions made by the model on both training and validation periods.
            - The black points are the actual values of the target on the training period.
            - The red line is the trend estimated by the model.
        """)

        # Prophet model and prediction
        model = Prophet()
        model.fit(df[['ds', 'y']])
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        # Plot Forecast with plotly
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast)

        # Performance Metrics Section
        st.markdown("## 2. Evaluation on validation set")
        st.markdown("### Performance metrics")

        # Explanation for the metrics
        with st.expander("More info on evaluation metrics"):
            st.markdown("""
            The following metrics can be computed to evaluate model performance:
            - **Mean Absolute Percentage Error (MAPE):** Measures the average absolute size of each error in percentage of the truth. 
            - **Symmetric Mean Absolute Percentage Error (SMAPE):** Slight variation of the MAPE, more robust to 0 values.
            - **Mean Squared Error (MSE):** Measures the average squared difference between forecasts and true values.
            - **Root Mean Squared Error (RMSE):** Square root of the MSE.
            - **Mean Absolute Error (MAE):** Measures the average absolute error.
            """)

        # Show Metric Formulas Option
        show_metric_formulas = st.checkbox("Show metric formulas")

        # Adjustments to match your design

        # Create dummy data to simulate the performance metrics over time
        dates = pd.date_range(start="2024-04-24", periods=30)
        performance_metrics = pd.DataFrame({
            "MAPE": abs(np.random.randn(30)) * 0.1,
            "RMSE": abs(np.random.randn(30)) * 50,
            "SMAPE": abs(np.random.randn(30)) * 0.1,
            "MAE": abs(np.random.randn(30)) * 20,
            "Date": dates
        })

        # Global Performance Metrics Display
        st.markdown("<h3 style='color: white;'>Global performance</h3>", unsafe_allow_html=True)
        cols = st.columns(5)
        metrics = {"RMSE": "19.31", "MAPE": "0.0696", "MAE": "14.16", "MSE": "372.8", "SMAPE": "0.0728"}
        for idx, (key, value) in enumerate(metrics.items()):
            with cols[idx]:
                st.markdown(f"<h1 style='color: white; font-weight: bold;'>{value}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #FC5296;'>{key}</p>", unsafe_allow_html=True)

        # Deep Dive Bar Plots with Color Updates
        st.markdown("<h3 style='color: white;'>Deep dive</h3>", unsafe_allow_html=True)

        # Plot each metric with individual plots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        plt.style.use('dark_background')
        colors = ['#1f3b5c', '#e21b5a', '#2cb8b8', '#ff993e']
        metrics_list = ["MAPE", "RMSE", "SMAPE", "MAE"]

        for i, metric in enumerate(metrics_list):
            ax = axs[i // 2, i % 2]
            sns.barplot(data=performance_metrics, x="Date", y=metric, ax=ax, color=colors[i])
            ax.set_title(metric)
            plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        st.pyplot(fig)
else:
    st.warning("Please upload a file to proceed.")
