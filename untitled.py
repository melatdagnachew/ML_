import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Set up the Streamlit app
st.title("Agricultural Exports Analysis")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write("Data Preview:")
    st.dataframe(df.head())
    if st.checkbox("Transpose Dataset"):
        if df.index.duplicated().any():
                st.warning("Duplicate years found. Adjusting indices.")
                df = df.reset_index()
    
        df = df.T
    # Check for missing values
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write(df.head())
    # Data Cleaning
    if st.button("Clean Data"):
        df_cleaned = df.dropna()  # Basic cleaning: drop missing values
        st.write("Cleaned Data Preview:")
        st.dataframe(df_cleaned.head())
    else:
        df_cleaned = df

    # Descriptive Statistics
    st.write("Descriptive Statistics:")
    st.write(df_cleaned.describe())


    # Ensure numeric data only for correlation
    numeric_df = df_cleaned.select_dtypes(include=[float, int])  # Keep only numeric columns

    if numeric_df.empty:
        st.write("No numeric columns available for correlation heatmap.")
    else:
        plt.figure(figsize=(10, 6))
        correlation_matrix = numeric_df.corr()  # Compute correlation only on numeric columns
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        st.pyplot(plt)
    # Regression Analysis
    # Regression Analysis
    if st.button("Run Regression"):
        
        X = df_cleaned[['Total Population (millions)',
                        'Rural population (% of total population)',
                        'Rural population growth (annual %)',
                        'Agricultural land (% of land area)',
                        'Fertilizer consumption (kg/ha of arable land)',
                        'Agriculture, value added (% of GDP)',
                        ]]  # Add more variables as needed
        y = df_cleaned['Agricultural exports, mln. US$']

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)


        # Calculate evaluation metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)

        # Display evaluation metrics
        st.subheader("Model Evaluation Metrics")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        st.write(f"**R-squared (R²):** {r2:.2f}")
        # Display the coefficients
        st.write("Model Coefficients:")
        for feature, coef in zip(X.columns, model.coef_):
            st.write(f"{feature}: {coef}")

        # Feature Importance Graph
        st.subheader("Feature Importance")

        # Create a DataFrame for coefficients
        coefficients_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        bars = plt.barh(coefficients_df['Feature'], coefficients_df['Coefficient'], 
                        color=['#1f77b4' if coef > 0 else '#ff7f0e' for coef in coefficients_df['Coefficient']],
                        alpha=0.8)

        # Add coefficient values as labels on the bars
        for bar in bars:
            plt.text(bar.get_width() + (0.02 if bar.get_width() > 0 else -0.02), 
                     bar.get_y() + bar.get_height() / 2, 
                     f'{bar.get_width():.2f}', 
                     va='center', 
                     fontsize=10, 
                     color='black')

        # Set titles and labels
        plt.xlabel('Coefficient Value', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title('Feature Importance from Linear Regression', fontsize=16)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)  # Line at x=0
        plt.gca().invert_yaxis()  # Reverse y-axis for better readability
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(plt)

     




            
