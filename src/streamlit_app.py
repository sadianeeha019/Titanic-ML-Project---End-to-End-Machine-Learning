import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = 'models/best_model.joblib'

st.set_page_config(page_title='Titanic Survival Predictor - Enhanced', page_icon='🚢')
st.title('🚢 Titanic Survival Prediction (Enhanced)')
st.write('Interactive demo using the best model saved by the Colab notebook.')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.warning('Best model not found. Run the enhanced Colab notebook and upload models/best_model.joblib to this repo.')
    model = None

with st.form('input_form'):
    pclass = st.selectbox('Pclass', [1,2,3], index=2)
    sex = st.selectbox('Sex', ['male','female'])
    age = st.slider('Age', 0, 80, 29)
    sibsp = st.number_input('SibSp', min_value=0, max_value=10, value=0)
    parch = st.number_input('Parch', min_value=0, max_value=10, value=0)
    fare = st.number_input('Fare', min_value=0.0, value=32.2)
    embarked = st.selectbox('Embarked', ['C','Q','S'])
    title = st.selectbox('Title', ['Mr','Mrs','Miss','Other'])
    submitted = st.form_submit_button('Predict')

if submitted:
    sample = {'Pclass':pclass, 'Sex': 0 if sex=='male' else 1, 'Age':age, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare, 'FamilySize': sibsp+parch+1, 'IsAlone': 1 if (sibsp+parch+1)==1 else 0}
    # add embarked and title dummies
    for col in ['Embarked_Q','Embarked_S','Title_Mrs','Title_Miss','Title_Other']:
        sample[col] = 0
    if embarked=='Q': sample['Embarked_Q'] = 1
    if embarked=='S': sample['Embarked_S'] = 1
    if title=='Mrs': sample['Title_Mrs'] = 1
    if title=='Miss': sample['Title_Miss'] = 1
    if title=='Other': sample['Title_Other'] = 1
    df = pd.DataFrame([sample])
    if model is None:
        st.error('Model not loaded.')
    else:
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else None
        if pred==1:
            st.success(f'✅ Survived with probability {prob:.2%}' if prob is not None else '✅ Survived')
        else:
            st.error(f'❌ Did not survive (prob {prob:.2%})' if prob is not None else '❌ Did not survive')