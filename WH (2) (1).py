#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().system('pip install openpyxl')
get_ipython().system('pip install plotly')
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px


# In[2]:


stat = pd.read_excel('Telco_customer_churn_status.xlsx')
demo = pd.read_excel('Telco_customer_churn_demographics.xlsx')
serv = pd.read_excel('Telco_customer_churn_services.xlsx')
loca = pd.read_excel('Telco_customer_churn_location.xlsx')

key = ['Customer ID']
df = demo.merge(
    loca, left_on=key, right_on=key).merge(
    serv, left_on=key, right_on=key).merge(
    stat, left_on=key, right_on=key)
df.head(5)


# In[3]:


df['Multiple Lines'].value_counts()


# In[4]:


to_drop = ['Count_x','Quarter_x','Customer Status','Offer','Churn Label','Churn Score','Churn Reason','Count_y','Under 30','Senior Citizen','Dependents','Count_x',
           'Quarter_y','Referred a Friend','Number of Referrals','Avg Monthly Long Distance Charges','Internet Type','Avg Monthly GB Download','Total Refunds',
           'Total Extra Data Charges','Total Long Distance Charges','Total Revenue','Count_y','Country','State','Zip Code','Lat Long']
df.drop(to_drop, axis=1, inplace=True)
df.head(5)


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


treatment = {'Churn Category': 'Not Churn'}
df = df.fillna(value=treatment)


# In[8]:


df.info()


# In[9]:


df.isna().sum()


# In[10]:


df['Number of Dependents'].unique()


# In[11]:


df['Age_Label'] = None
df['Churn_Label']=None
df['Month_Label']=None


# In[12]:


df.loc[df['Age'] < 30, 'Age_Label'] = 'Under 30 years'
df.loc[df['Age'] > 65, 'Age_Label'] = 'Over 65 years'
df.loc[(df['Age']>=30) & (df['Age']<=65), 'Age_Label'] = 'From 30 to 65 years'
df.loc[df['Churn Value'] ==1, 'Churn_Label'] = 'Đã rời bỏ'
df.loc[df['Churn Value'] ==0, 'Churn_Label'] = 'Chưa rời bỏ'
df.loc[df['Tenure in Months'] <=3, 'Month_Label'] = '1 - 3 tháng'
df.loc[(df['Tenure in Months'] <=6) & (df['Tenure in Months'] >3), 'Month_Label'] = '3 - 6 tháng'
df.loc[(df['Tenure in Months'] <=12) & (df['Tenure in Months'] >6), 'Month_Label'] = '6 tháng - 1 năm'
df.loc[(df['Tenure in Months'] <=24) & (df['Tenure in Months'] >12), 'Month_Label'] = '1 - 2 năm'
df.loc[(df['Tenure in Months'] <=60) & (df['Tenure in Months'] >24), 'Month_Label'] = '2 - 5 năm'
df.loc[(df['Tenure in Months'] >60), 'Month_Label'] = 'trên 5 năm'
df['Number of Dependents'].replace([0,1,2,3,4,5,6,7,8,9], ['No','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes'], inplace=True)
df.head(5)


# In[13]:


df.info()


# In[15]:


export_csv = df.to_csv (r'Telco customer churn.csv', index = None, header=True)


# In[ ]:


df['Month_Label'].unique()


# In[ ]:


df['Multiple Lines'].value_counts()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, ConfusionMatrixDisplay, r2_score


# In[ ]:


z=X.describe()
z


# In[ ]:


sc = StandardScaler()
X= sc.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


clf_name = []
roc_auc = []
f1 = []
def model_eval(clf, y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cm=confusion_matrix(y_test, y_pred, labels=y_test.unique())
    disp = ConfusionMatrixDisplay(cm, display_labels=y_test.unique())
    disp.plot(cmap='cividis')
    m1 = roc_auc_score(y_test, y_pred)
    m2 = f1_score(y_test, y_pred)
    print('ROC_AUC_Score: {:.04f}'.format(m1))
    print('F1 Score: {:.04f}'.format(m2))
    clf_name.append(clf)
    roc_auc.append(m1)
    f1.append(m2)


# In[ ]:


xgb_clf = XGBClassifier(n_estimators=500, max_depth=1, max_leaves=2, random_state=0)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
clf = 'XGBoost'
model_eval(clf, y_test, y_pred)


# ## Tenure

# In[ ]:


y = model_df['Tenure in Months']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_test[1]


# In[ ]:


xgb_r = XGBRegressor(n_estimators=1000, random_state=42)
xgb_r.fit(X_train,y_train)
y_pred = xgb_r.predict(X_test)
print(f'R Squared Score: {r2_score(y_pred, y_test)}')


# ## Predictions on a Custom User Data

# In[ ]:


def TEST(test,z):
    listtest=[]
    L={'c1':['Age','Monthly Charge','Total Charges'],
      'c2':['Number of Dependents','Married','Phone Service','Multiple Lines','Internet Service','Online Security','Online Backup','Paperless Billing',
            'Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data']}
    for k in test:
        if k in L['c1']:
            x=(test[k]-z[k][1])/z[k][2]
            listtest.append(x)
        elif k in L['c2']:
            a=str(k)+'_Yes'
            b=str(k)+'_No'
            if test[k]=='Yes':
                x1=(0-z[b][1])/z[b][2]
                x2=(1-z[a][1])/z[a][2]
            else:
                x1=(1-z[b][1])/z[b][2]
                x2=(0-z[a][1])/z[a][2]
            listtest.append(x1)
            listtest.append(x2)
        elif k=='Gender':
            a=str(k)+'_Female'
            b=str(k)+'_Male'
            if test[k]=='Female':
                x1=(1-z[a][1])/z[a][2]
                x2=(0-z[b][1])/z[b][2]
            else:
                x1=(0-z[a][1])/z[a][2]
                x2=(1-z[b][1])/z[b][2]
            listtest.append(x1)
            listtest.append(x2)
        elif k=='Contract':
            for i in ['Month-to-Month','One Year','Two Year']:
                a=str(k)+'_'+str(i)
                if test[k]==i:
                    x=(1-z[a][1])/z[a][2]
                else:
                    x=(0-z[a][1])/z[a][2]
                listtest.append(x)
        elif k=='Payment Method':
            for i in ['Bank Withdrawal','Credit Card','Mailed Check']:
                a=str(k)+'_'+str(i)
                if test[k]==i:
                    x=(1-z[a][1])/z[a][2]
                else:
                    x=(0-z[a][1])/z[a][2]
                listtest.append(x)
    import numpy as np
    user_input=np.array([listtest])
    return user_input


# In[ ]:


test={'Age':78,'Monthly Charge':39,'Total Charges':165,'Gender':'Male','Married':'No',
      'Number of Dependents':'No','Phone Service':'No','Multiple Lines':'No','Internet Service':'Yes',
      'Online Security':'No','Online Backup':'No','Device Protection Plan':'No','Premium Tech Support':'No',
      'Streaming TV':'No','Streaming Movies':'No','Streaming Music':'No','Unlimited Data':'No','Contract':'Month-to-Month',
      'Paperless Billing':'No','Payment Method':'Credit Card'}


# In[ ]:


churn = ['No', 'Yes']
print('Số tháng khách hàng sẽ ở lại với cty:',xgb_r.predict(TEST(test,z))[0])
print('KH có rời bỏ không?',churn[xgb_clf.predict(TEST(test,z))[0]])
print('xác suất khách hàng ở lại công ty là:',(100-xgb_clf.predict_proba(TEST(test,z))[:, 1][0]*100))


# In[ ]:


test={'Age':30,'Monthly Charge':50,'Total Charges':3640,'Gender':'Male','Married':'Yes',
      'Number of Dependents':'Yes','Phone Service':'No','Multiple Lines':'No','Internet Service':'Yes',
      'Online Security':'No','Online Backup':'No','Device Protection Plan':'No','Premium Tech Support':'No',
      'Streaming TV':'No','Streaming Movies':'No','Streaming Music':'No','Unlimited Data':'No','Contract':'Month-to-Month',
      'Paperless Billing':'No','Payment Method':'Credit Card'}


# In[ ]:


print('Số tháng khách hàng sẽ ở lại với cty:',xgb_r.predict(TEST(test,z))[0])
print('KH có rời bỏ không?',churn[xgb_clf.predict(TEST(test,z))[0]])
print('xác suất khách hàng ở lại công ty là:',(100-xgb_clf.predict_proba(TEST(test,z))[:, 1][0]*100))


# In[ ]:


# model_df.replace(['Đã rời bỏ','Chưa rời bỏ'], ['Churned','Not Churn'], inplace=True)
# model_df


# In[ ]:


# export_csv = model_df.to_csv (r'Telco_customer_churn.csv', index = None, header=True)


# In[ ]:




