import streamlit as st
import eda
import prediction

navigation = st.sidebar.radio('Choose  : ', ('EDA', 'Prediction'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()
