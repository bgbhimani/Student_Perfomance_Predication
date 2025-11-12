# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Data Processing
df = pd.read_csv("Student_Performance.csv")
X = df.drop(columns=['Performance Index'])
y = df['Performance Index']
non_binary_cols = ['Hours Studied', 'Previous Scores',  'Sleep Hours', 'Sample Question Papers Practiced']
binary_cols = ['Extracurricular Activities']
scaler = StandardScaler()
if df['Extracurricular Activities'].dtype == object:
	df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
df['Extracurricular Activities'] = pd.to_numeric(df['Extracurricular Activities'], errors='coerce')
df = df.dropna(subset=non_binary_cols + binary_cols + ['Performance Index'])
X_scaled_part = scaler.fit_transform(df[non_binary_cols])
X_scaled = np.column_stack([X_scaled_part, df['Extracurricular Activities'].values])
X = pd.DataFrame(X_scaled,columns=non_binary_cols + binary_cols)
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=14)

# Training Models:

# 1. Scikit-learn 
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)

# 2. Custom Multiple Linear Regression
class MLR:
    def __inti__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0,1,axis=1)

        betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
            
    def predict(self, X_test):
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
    
mlr = MLR()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)

# 3. Gradient Descent Method
class GDRegressor:
        def __init__(self,learning_rate=0.001,epochs=100): 
            self.intercept_ = None
            self.coef_ = None
            self.lr = learning_rate
            self.epochs = epochs

        def fit(self,X_train,y_train):
            self.intercept_ = 0
            self.coef_ = np.ones(X_train.shape[1])

            for i in range(self.epochs):
                # update all intercept And coef_
                yhat = np.dot(X_train, self.coef_) + self.intercept_
                # print(yhat.shape)
                
                intercept_der = -2*np.mean(y_train - yhat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der)

                coef_der = -2 * np.dot((y_train-yhat),X_train) / X_train.shape[0]
                self.coef_ = self.coef_ - (self.lr * coef_der)
            # print(self.intercept_, self.coef_)

                
        def predict(self,X_test):
            return np.dot(X_test,self.coef_) + self.intercept_
gdr = GDRegressor(epochs=80,learning_rate=0.1)
gdr.fit(X_train,y_train)
y_pred_gdr = gdr.predict(X_test)





## Stremalit App
st.set_page_config(
    page_title="My Wide Streamlit App",
    layout="wide",
    initial_sidebar_state="expanded"
)
Description = "Description: Implements Multiple Linear Regression from scratch (closed-form normal equation) and a batch Gradient Descent regressor. The code preprocesses data (maps binary 'Extracurricular Activities' to 0/1 and standard-scales continuous features), fits custom models alongside scikit-learn's LinearRegression, and evaluates performance with R², MAE, and MSE."
st.title("Student Performance Prediction")
st.write(Description)
st.badge("Model gets trained. Please wait...", color="yellow")

non_binary_cols = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']
binary_cols = ['Extracurricular Activities']

col1, col2 = st.columns(2)
with col1:
    hours_studied = st.number_input("Enter Hours Studied: ",step = 1)
    previous_scores = st.number_input("Enter Previous Scores: ",step = 1)
    sleep_hours = st.number_input("Enter Sleep Hours: ",step = 1)
    
with col2:
    sample_papers = st.number_input("Enter Sample Question Papers Practiced: ",step = 1)
    # extra_activities = st.number_input("Extracurricular Activities (0 = No, 1 = Yes): ")
    extra_activities = st.selectbox(label="Extracurricular Activities",options=['True','False'])
    if extra_activities == 'True':
        extra_activities = 1
    else:
        extra_activities = 0
    btn1 = st.button("Predict the Score")

if btn1:
    X_new_non_binary = np.array([[hours_studied, previous_scores, sleep_hours, sample_papers]])
    X_new_non_binary_scaled = scaler.transform(X_new_non_binary)
    X_new_full = np.column_stack([X_new_non_binary_scaled, np.array([[extra_activities]])])
    y_new_pred_mlr = mlr.predict(X_new_full)
    st.header(f"Predicted Performance Score: {y_new_pred_mlr[0]:.2f}")

st.space(50)
st.title("Model Performance Comparision")

score = pd.DataFrame(
    {
        "scikit-learn": [r2_score(y_test,y_pred_lr)*100,mean_absolute_error(y_test, y_pred_lr),mean_squared_error(y_test, y_pred_lr)],
        "Custom MLR":[r2_score(y_test,y_pred_mlr)*100,mean_absolute_error(y_test, y_pred_mlr),mean_squared_error(y_test, y_pred_mlr)],
        "Gradient Descent":[r2_score(y_test,y_pred_gdr)*100,mean_absolute_error(y_test, y_pred_gdr),mean_squared_error(y_test, y_pred_gdr)]
    },
    index=["r2 score","MAE","MSE"]
)
st.table(score)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)




st.space(50)
st.title("Multiple Linear Regression Using the Normal Equation (Custom Class)")
st.text("The Custom Class for the Multiple Linear Regresion:")
st.code("""
class MLR:
    def __inti__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        X_train = np.insert(X_train,0,1,axis=1)

        betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
            
    def predict(self, X_test):
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
            """,language="python")

st.space(50)
st.text("Multiple Linear Regression Using the Batch Gradient Descent Method (Custom Class)")
st.code("""
class GDRegressor:
    def __init__(self,learning_rate=0.001,epochs=100): 
        self.intercept_ = None
        self.coef_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # update all intercept And coef_
            yhat = np.dot(X_train, self.coef_) + self.intercept_
            # print(yhat.shape)
            
            intercept_der = -2*np.mean(y_train - yhat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)

            coef_der = -2 * np.dot((y_train-yhat),X_train) / X_train.shape[0]
            self.coef_ = self.coef_ - (self.lr * coef_der)
        print(self.intercept_, self.coef_)

            
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_""",language="python")
