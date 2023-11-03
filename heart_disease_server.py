#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
model = pickle.load(open('./heart_disease_model.pkl', 'rb'))



# In[5]:


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach,oldpeak, ca, thal):
    prediction=model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,oldpeak, ca, thal]])
    return prediction

def main():
    st.title("HEART DISEASE PREDICTION")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit HEART DISEASE PREDICTION</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input(label="age",placeholder="Enter age in number")
    sex = st.text_input(label="Sex",placeholder="Enter 1 for Male and 0 for Female")
    cp = st.text_input(label="Chest pain type",placeholder="Enter 0 for typical angina, 1 for atypical angina, 2 for non-anginal pain, 3 for asymptomatic")
    trestbps = st.text_input(label="Resting Blood Pressure",placeholder="Enter in numbers unit is mm Hg")
    chol=st.text_input(label="Serum Cholestrol",placeholder="Enter in number mg/dl")
    fbs=st.text_input(label="Fasting blood sugar",placeholder="Enter 1 if fbs>120 mg/dl enter 0 otherwise")
    restecg=st.text_input(label="resting electrocardiographic results",placeholder="Enter 0 for normal, 1 for ST-T wave abnormility, 2 for left ventricular hypertrophy")
    thalach=st.text_input(label="maximum heart rate",placeholder="Enter in number")
    oldpeak=st.text_input(label="ST depression induced by exercise",placeholder="Enter number")
    ca=st.text_input(label="number of vessels", placeholder="Enter value between 0 and 3")
    thal=st.text_input(label="thal",placeholder="0 for normal, 1 for fixed defect, 2 for reversable defect")
    result=""
    if st.button("Predict"):
        result=predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach,oldpeak, ca, thal)
    if result==0:
        st.success('You don\'t have a heart condition')
    elif result==1:
        st.success('You have a heart condition')
    else:
        st.success("waiting for input")
    
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
main()


