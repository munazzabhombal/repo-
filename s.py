import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


df = pd.read_csv("practo_cleandata.csv")
st.title("Doctor Fee Prediction")

for column in df.columns:
  print(df[column].value_counts())
  print("-"*20)

degree_count = df['Degree'].value_counts()
degree_count_less_10 = degree_count[degree_count <=10]
df['Degree'] = df['Degree'].apply(lambda x: 'other' if x in degree_count_less_10 else x)

Location_count = df['Location'].value_counts()
Location_count_less_10 = Location_count[Location_count <=10]
df['Location'] = df['Location'].apply(lambda x: 'other' if x in Location_count_less_10 else x)

df.drop(columns='Name', inplace=True)

X= df.drop(columns='Fees')
Y = df['Fees']

def predict(values_to_predict):
	X_train,X_test,Y_train,Y_test =train_test_split(X,Y, test_size=0.2,random_state=45)
	column_trans = make_column_transformer((OneHotEncoder(sparse=False),['Degree','Specialization','Location','City']))
	scaler = StandardScaler()
	lr = LinearRegression(normalize=True)
	pipe =make_pipeline(column_trans,scaler,lr)
	pipe.fit(X_train,Y_train)
	Y_pred_lr= pipe.predict(values_to_predict)
	return Y_pred_lr

speciality = st.selectbox(
         label="Select Specialization", options=df["Specialization"].unique())

degree = st.selectbox(
       label="Select Degree", options=df["Degree"].unique())

experience = st.number_input("Years of Experience:", step=1)


location = st.selectbox(
       label="Select Location", options=df["Location"].unique())


city = st.selectbox(
         label="Select City", options=df["City"].unique())

dp_score = st.number_input("DP Score:", step=1)

npv = st.number_input("NPV:", step=1)

value = [[speciality, degree, experience, location, city, dp_score, npv]]
values_to_predict = np.array(value).reshape(1,-1)


if st.button("Calculate Fee"):
  # Calculate consultation fee based on the given values
  
  st.write(f"Doctor Fee: ₹ {predict(values_to_predict)}")