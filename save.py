import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Streamlit App Title
st.title("Agricultural Sustainability Predictor")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    if st.checkbox("Transpose Dataset"):
        if data.index.duplicated().any():
                st.warning("Duplicate years found. Adjusting indices.")
                data = data.reset_index()
    
        data = data.T
    
                
    # Display dataset overview
    st.subheader("Dataset Overview")
    st.write(data.head())
    st.write(data.columns)
    # Handle missing values
    st.subheader("Handling Missing Values")
    if st.checkbox("Interpolate Missing Values"):
        data = data.interpolate(method='linear')
        st.write("Linear interpolation applied.")
    if st.checkbox("Fill Missing Values with Mean"):
        data = data.fillna(data.mean())
        st.write("Missing values filled with column mean.")

    # Feature Engineering
    st.subheader("Feature Engineering")
    try:
        data['Population Growth Rate'] = data['Total Population (millions)'].pct_change().fillna(0)
        data['Agricultural Yield per Capita'] = data['Agricultural exports, mln. US$'] / data['Total Population (millions)']
        data['Arable Land per Capita'] = (
            (data['Arable Land (% of Land Area)'] / 100) *
            data['Land area (1000 sq. km)'] /
            data['Total Population (millions)']
        )
        data['Agricultural Yield Growth Rate'] = data['Agricultural Yield per Capita'].pct_change().fillna(0)
        data['Sustainability'] = np.where(
            data['Agricultural Yield Growth Rate'] >= data['Population Growth Rate'],
            1, 0
        )
        st.write("Feature engineering completed.")
    except KeyError as e:
        st.error(f"Missing columns required for feature engineering: {e}")

    # Feature Selection
    features = [
        'Total Population (millions)',
        'Rural population (% of total population)',
        'Rural population growth (annual %)',
        'Agricultural land (% of land area)',
        'Arable Land per Capita',
        'Fertilizer consumption (kg/ha of arable land)',
        'Agricultural Yield per Capita',
        'Population Growth Rate',
    ]
    available_features = [feature for feature in features if feature in data.columns]
    if not available_features:
        st.error("No suitable features found in the dataset.")
    else:
        st.subheader("Model Training and Evaluation")
        X = data[available_features]
        y = data['Sustainability']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)

        # Display Metrics
        st.write("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        # Feature Importance
        st.subheader("Feature Importances")
        importances = clf.feature_importances_
        feature_importances = pd.Series(importances, index=available_features).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importances.plot(kind='bar', color='skyblue')
        plt.title("Feature Importances")
        plt.ylabel("Importance")
        st.pyplot(plt)
