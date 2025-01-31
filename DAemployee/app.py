import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Define the columns to be encoded
encoding_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']

# Load the data
@st.cache_resource
def load_data():
    data = pd.read_csv('/content/Employee-Attrition.csv')
    data.drop_duplicates(inplace=True)
    data['Attrition'] = data['Attrition'].replace({'No': 0, 'Yes': 1})
    data['OverTime'] = data['OverTime'].map({'No': 0, 'Yes': 1})
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    label_encoders = {}
    for column in encoding_cols:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data, label_encoders

data, label_encoders = load_data()

X = data.drop(['Attrition', 'Over18'], axis=1)
y = data['Attrition'].values

rus = RandomOverSampler(random_state=42)
X_over, y_over = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Define prediction function
def predict_employee_attrition(model, label_encoders, employee_input):
    employee_df = pd.DataFrame([employee_input])
    for column in encoding_cols:
        if column in employee_df.columns and column in label_encoders:
            employee_df[column] = label_encoders[column].transform(employee_df[column])
    binary_columns = {
        'OverTime': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 0, 'Female': 1}
    }
    for column, mapping in binary_columns.items():
        if column in employee_df.columns:
            employee_df[column] = employee_df[column].map(mapping)
    employee_df = employee_df[X.columns]
    prediction = model.predict(employee_df)
    prediction_proba = model.predict_proba(employee_df)[:, 1]
    return prediction[0], prediction_proba[0]

# Streamlit app
st.title("Employee Attrition Prediction")

st.sidebar.header("Employee Input Features")

def user_input_features():
    Age = st.sidebar.slider('Age', 18, 60, 28)
    BusinessTravel = st.sidebar.selectbox('BusinessTravel', ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))
    DailyRate = st.sidebar.number_input('DailyRate', value=1000)
    Department = st.sidebar.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))
    DistanceFromHome = st.sidebar.slider('DistanceFromHome', 1, 30, 10)
    Education = st.sidebar.selectbox('Education', [1, 2, 3, 4, 5])
    EducationField = st.sidebar.selectbox('EducationField', ('Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'))
    EmployeeCount = 1
    EmployeeNumber = st.sidebar.number_input('EmployeeNumber', value=9999)
    EnvironmentSatisfaction = st.sidebar.selectbox('EnvironmentSatisfaction', [1, 2, 3, 4])
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    JobInvolvement = st.sidebar.selectbox('JobInvolvement', [1, 2, 3, 4])
    JobLevel = st.sidebar.selectbox('JobLevel', [1, 2, 3, 4, 5])
    JobRole = st.sidebar.selectbox('JobRole', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
    JobSatisfaction = st.sidebar.selectbox('JobSatisfaction', [1, 2, 3, 4])
    MaritalStatus = st.sidebar.selectbox('MaritalStatus', ('Single', 'Married', 'Divorced'))
    MonthlyIncome = st.sidebar.number_input('MonthlyIncome', value=50000)
    MonthlyRate = st.sidebar.number_input('MonthlyRate', value=15000)
    NumCompaniesWorked = st.sidebar.slider('NumCompaniesWorked', 0, 10, 1)
    OverTime = st.sidebar.selectbox('OverTime', ('No', 'Yes'))
    PercentSalaryHike = st.sidebar.slider('PercentSalaryHike', 0, 25, 15)
    PerformanceRating = st.sidebar.selectbox('PerformanceRating', [1, 2, 3, 4])
    RelationshipSatisfaction = st.sidebar.selectbox('RelationshipSatisfaction', [1, 2, 3, 4])
    StockOptionLevel = st.sidebar.selectbox('StockOptionLevel', [0, 1, 2, 3])
    TotalWorkingYears = st.sidebar.slider('TotalWorkingYears', 0, 40, 10)
    TrainingTimesLastYear = st.sidebar.slider('TrainingTimesLastYear', 0, 10, 3)
    WorkLifeBalance = st.sidebar.selectbox('WorkLifeBalance', [1, 2, 3, 4])
    YearsAtCompany = st.sidebar.slider('YearsAtCompany', 0, 40, 5)
    YearsInCurrentRole = st.sidebar.slider('YearsInCurrentRole', 0, 20, 2)
    YearsSinceLastPromotion = st.sidebar.slider('YearsSinceLastPromotion', 0, 20, 1)
    YearsWithCurrManager = st.sidebar.slider('YearsWithCurrManager', 0, 20, 1)
    HourlyRate = st.sidebar.slider('HourlyRate', 0, 100, 50)
    StandardHours = 80

    data = {
        'Age': Age,
        'BusinessTravel': BusinessTravel,
        'DailyRate': DailyRate,
        'Department': Department,
        'DistanceFromHome': DistanceFromHome,
        'Education': Education,
        'EducationField': EducationField,
        'EmployeeCount': EmployeeCount,
        'EmployeeNumber': EmployeeNumber,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'Gender': Gender,
        'JobInvolvement': JobInvolvement,
        'JobLevel': JobLevel,
        'JobRole': JobRole,
        'JobSatisfaction': JobSatisfaction,
        'MaritalStatus': MaritalStatus,
        'MonthlyIncome': MonthlyIncome,
        'MonthlyRate': MonthlyRate,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': OverTime,
        'PercentSalaryHike': PercentSalaryHike,
        'PerformanceRating': PerformanceRating,
        'RelationshipSatisfaction': RelationshipSatisfaction,
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'HourlyRate': HourlyRate,
        'StandardHours': StandardHours
    }
    return data

employee_input = user_input_features()

prediction, prediction_proba = predict_employee_attrition(logreg, label_encoders, employee_input)

# Make the prediction output more beautiful
st.subheader('Prediction')

if prediction == 1:
    st.markdown(
        f"<h2 style='text-align: center; color: red;'>Leave</h2>",
        unsafe_allow_html=True
    )
    st.image("https://cms-assets.recognizeapp.com/wp-content/uploads/2022/02/23070236/how-do-you-motivate-an-unhappy-employee.webp", use_column_width=True)
else:
    st.markdown(
        f"<h2 style='text-align: center; color: green;'>Stay</h2>",
        unsafe_allow_html=True
    )
    st.image("https://ekosnegocios.com/image/posts/header/10876.jpg", use_column_width=True)

st.write(f"Probability of leaving: {prediction_proba:.2f}")
