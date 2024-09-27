from pickle import load
import streamlit as st


model = load(open("models/model_xgb.pkl", "rb"))

class_dict = {"1":'Approved', "0":'Not Approved'}


st.title("Student Approval Prediction")


val1 = st.slider("AttendanceRate", min_value = 10, max_value = 100, step = 1)

val2 = st.slider("StudyHoursPerWeek", min_value = 10, max_value = 80, step = 1)

val3 = st.slider("PreviousGrade", min_value = 0.5, max_value = 10.0, step = 0.01)

val4 = st.slider("ExtracurricularActivities", min_value = 100, max_value = 150, step = 1)


if st.button("Predict"):

    prediction = (model.predict_proba([[val1, val2, val3, val4]])[:, 1])
    prob= (prediction[0]*100).round(2)

    st.write("Prediction:", prob, "%" " of probability Approval")





