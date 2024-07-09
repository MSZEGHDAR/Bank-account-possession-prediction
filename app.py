import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Bank account possession prediction")

fields=['country', 'year', 'location_type', 'cellphone_access',
       'household_size', 'age_of_respondent', 'gender_of_respondent',
       'relationship_with_head', 'marital_status', 'education_level',
       'job_type']



model = joblib.load('model.joblib')

country=st.selectbox('Country',['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
year=st.slider('Year',2016,2018)
location_type=st.selectbox('Location Type',['Rural', 'Urban'])
cellphone_access=st.selectbox('Cellphone Access',[1, 0])
household_size=st.slider('Household Size',1,20)
age_of_respondent=st.slider('Age of respondent',24,100)
gender_of_respondent=st.selectbox('Gender of respondent',['Male', 'Female'])
relationship_with_head=st.selectbox('Relationship with head',['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
       'Other non-relatives'])
marital_status=st.selectbox('Marital Status',['Married/Living together', 'Widowed', 'Single/Never Married',
       'Divorced/Seperated', 'Dont know'])
education_level=st.selectbox('Education Level',['Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA'])
job_type=st.selectbox('job_type',['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])


if st.button('Predict'):
       input_data = pd.DataFrame([[country, location_type, cellphone_access, gender_of_respondent, 
                                   relationship_with_head, marital_status, education_level, job_type, 
                                   year, household_size, age_of_respondent]], 
                                   columns=['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                                          'relationship_with_head', 'marital_status', 'education_level', 'job_type',
                                          'year', 'household_size', 'age_of_respondent'])


       numeric_fields = ['year', 'household_size', 'age_of_respondent']
       for field in numeric_fields:
              input_data[field] = input_data[field].astype(float)


       categorical_fields = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent', 
                            'relationship_with_head', 'marital_status', 'education_level', 'job_type']

       for field in categorical_fields:
              input_data[field] = input_data[field].astype(str)
    

       prediction = model.predict(input_data)
       proba=model.predict_proba(input_data)

       if prediction[0] == 1:
              st.write(f"The customer is likely to have a bank account with a probablity of {proba[0][1]:.1%}")
       else:
              st.write(f"The customer is unlikely to have a bank account with a probablity of {proba[0][0]:.1%}")
