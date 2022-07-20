import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
import random
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler


#应用标题
st.set_page_config(page_title='Pred BM in PCa')
st.title('Application of Machine Learning Methods to Predict Bone Metastases in Prostate Cancer Patients')



# conf
st.sidebar.markdown('## Variables')
Age = st.sidebar.selectbox('Age',('<=39','40-49','50-59','60-69','>=70'),index=4)
Race = st.sidebar.selectbox("Race",('American Indian/Alaska Native','Asian or Pacific Islander','Black','White'),index=3)
Grade = st.sidebar.selectbox("Grade",('Ⅰ','Ⅱ','Ⅲ','Ⅳ'),index=1)
T_stage = st.sidebar.selectbox("T stage",('T1','T2','T3','T4'))
N_stage = st.sidebar.selectbox("N stage",('N0','N1'))
Gleason_score = st.sidebar.selectbox("Gleason score",('<=6','7','8','>=9'))
PSA = st.sidebar.slider("PSA(ng/ml)", 0.1, 98.0, value=10.0, step=0.1)
Marital_status = st.sidebar.selectbox("Marital status",('Married','Unmarried'))

st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact liuwencaincu@163.com, Liu Wencai, Nanchang university')
# str_to_int

map = {'<=39':0,'40-49':1,'50-59':2,'60-69':3,'>=70':4,'American Indian/Alaska Native':1,'Asian or Pacific Islander':2,'Black':3,
       'White':4,'Ⅰ':1,'Ⅱ':2,'Ⅲ':3,'Ⅳ':4,'T1':1,'T2':2,'T3':3,'T4':4,'N0':0,'N1':1,'<=6':1,'7':2,'8':3,'>=9':4,
       'Married':1,'Unmarried':0}
Age =map[Age]
Race =map[Race]
Grade =map[Grade]
T_stage =map[T_stage]
N_stage =map[N_stage]
Gleason_score =map[Gleason_score]
Marital_status =map[Marital_status]

# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
thyroid_train['BM'] = thyroid_train['BM'].apply(lambda x : +1 if x==1 else 0)
features = ['Age','Race','Grade','T_stage','N_stage','PSA','Gleason_score','Marital_status']
target = 'BM'

ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

XGB = XGBClassifier(random_state=32,max_depth=3,n_es
#读存储的模型
#with open('XGB.pickle', 'rb') as f:
#    XGB = pickle


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[Age,Race,Grade,T_stage,N_stage,PSA,Gleason_score,Marital_status]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[Age,Race,Grade,T_stage,N_stage,PSA,Gleason_score,Marital_status]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for Bone Metastasis:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of High risk group








