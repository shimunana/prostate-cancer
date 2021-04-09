import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#应用标题
st.title('Application of Machine Learning Methods to Predict Bone Metastases in Prostate Cancer Patients')



# conf
st.sidebar.markdown('## Variables')
Age = st.sidebar.selectbox('Age',('<70','>=70'),index=1)
Race = st.sidebar.selectbox("Race",('American Indian/Alaska Native','Asian or Pacific Islander','Black','White'),index=3)
Grade = st.sidebar.selectbox("Grade",('Ⅰ','Ⅱ','Ⅲ','Ⅳ'),index=1)
T_stage = st.sidebar.selectbox("T stage",('T1','T2','T3','T4'))
N_stage = st.sidebar.selectbox("N stage",('N0','N1'))
Gleason_score = st.sidebar.selectbox("Gleason score",('<=6','7','8','>=9'))
PSA = st.sidebar.slider("PSA(ng/ml)", 0.1, 98.0, value=10.0, step=0.1)
Marital_status = st.sidebar.selectbox("Marital status",('Married','Unmarried'))

# str_to_int

map = {'<70':0,'>=70':1,'American Indian/Alaska Native':1,'Asian or Pacific Islander':2,'Black':3,
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
thyroid_test = pd.read_csv('test.csv', low_memory=False)
thyroid_test['BM'] = thyroid_test['BM'].apply(lambda x : +1 if x==1 else 0)
features = ['Age','Race','Grade','T_stage','N_stage','PSA','Gleason_score','Marital_status']
target = 'BM'

#train and predict
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=7,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
#RF.fit(thyroid_train[features],thyroid_train[target])
XGB = XGBClassifier(random_state=32,max_depth=3,n_estimators=34)
XGB.fit(thyroid_train[features],thyroid_train[target])
#读之前存储的模型

#with open('RF.pickle', 'rb') as f:
#    RF = pickle.load(f)


sp = 0.037
#figure
is_t = (XGB.predict_proba(np.array([[Age,Race,Grade,T_stage,N_stage,PSA,Gleason_score,Marital_status]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[Age,Race,Grade,T_stage,N_stage,PSA,Gleason_score,Marital_status]]))[0][1])#*1000//1/1000

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))
if is_t:
    result = 'Bone Metastasis'
else:
    result = 'Without Bone Metastasis'
st.markdown('## Predict:  '+str(result))
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



st.title("")
st.title("")
#st.warning('This is a warning')
#st.error('This is an error')
map_data = pd.DataFrame(
    [[28.65, 115.79]],
    columns=['lat', 'lon'])

st.info('Information of the model: Auc: 0.9546 ;Accuracy: 0.8883 ;Sensitivity(recall): 0.8975 ;Specificity :0.8880 ')
st.success('Affiliation: The First Affiliated Hospital of Nanchang University, Nanchnag university. ')
if st.button('Click for location'):
    st.balloons()
    st.map(map_data)




