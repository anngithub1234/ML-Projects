import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
st.title("Wine Quality Prediction")
st.write("This app predicts whether the wine is of the best quality based on various features.")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    dt = pd.read_csv(uploaded_file)
    
    # Data preprocessing
    st.subheader("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.write(dt.head(5))
    
    # Handling missing values
    for col in dt.columns:
        if dt[col].isnull().sum() > 0:
            dt[col] = dt[col].fillna(dt[col].mean())
    
    # Drop unnecessary columns
    if 'total sulfur dioxide' in dt.columns:
        dt = dt.drop('total sulfur dioxide', axis=1)
    
    # Adding a target column
    dt['best quality'] = [1 if x > 5 else 0 for x in dt['quality']]
    dt.replace({'white': 1, 'red': 0}, inplace=True)
    
    # Check for class imbalance
    class_counts = dt['best quality'].value_counts()
    st.write("Class distribution:")
    st.bar_chart(class_counts)
    
    # Splitting features and target
    features = dt.drop(['quality', 'best quality'], axis=1)
    target = dt['best quality']
    
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)
    
    # Handling missing values and scaling
    imputer = SimpleImputer(strategy='mean')
    xtrain = imputer.fit_transform(xtrain)
    xtest = imputer.transform(xtest)
    
    norm = MinMaxScaler()
    xtrain = norm.fit_transform(xtrain)
    xtest = norm.transform(xtest)
    
    # Model Training
    models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
    model_names = ["Logistic Regression", "XGBoost Classifier", "Support Vector Machine"]
    
    model_selection = st.selectbox("Select a Model for Prediction", model_names)
    selected_model = models[model_names.index(model_selection)]
    selected_model.fit(xtrain, ytrain)
    
    # Model Evaluation
    ytest_pred = selected_model.predict(xtest)  # Predict on test data
    cm = metrics.confusion_matrix(ytest, ytest_pred)
    st.subheader("Confusion Matrix")
    st.write(cm)
    
    # Visualizing the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
    
    # Input section for prediction
    st.subheader("Make a Prediction")
    st.write("Enter values for each feature below:")
    
    user_input = {}
    for feature in features.columns:
        if feature == "type":
            user_input[feature] = st.selectbox("Type of wine (red/white)", options=["red", "white"])
            user_input[feature] = 1 if user_input[feature] == "white" else 0
        else:
            user_input[feature] = st.number_input(f"{feature}", value=float(features[feature].mean()))
    
    if st.button("Predict"):
        # Convert input to a dataframe
        input_df = pd.DataFrame([user_input])
        
        # Apply preprocessing to the input
        input_df = imputer.transform(input_df)
        input_df = norm.transform(input_df)
        
        # Make a prediction
        prediction = selected_model.predict(input_df)
        result = "Best Quality" if prediction[0] == 1 else "Not Best Quality"
        
        st.subheader("Prediction Result")
        st.write(f"The wine is predicted to be: **{result}**")