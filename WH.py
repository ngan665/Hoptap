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
df.head()


# In[3]:


df['Multiple Lines'].value_counts()


# In[4]:


to_drop = ['Count_x','Quarter_x','Customer Status','Churn Label','Churn Score','Churn Reason','Count_y','Under 30','Senior Citizen','Dependents','Count_x',
           'Quarter_y','Referred a Friend','Number of Referrals','Avg Monthly Long Distance Charges','Internet Type','Avg Monthly GB Download','Total Refunds',
           'Total Extra Data Charges','Total Long Distance Charges','Total Revenue','Count_y','Country','State','Zip Code','Lat Long','Latitude','Longitude']
df.drop(to_drop, axis=1, inplace=True)
df


# In[5]:


df.info()


# In[6]:


treatment = {'Churn Category': 'Not Churn'}
df = df.fillna(value=treatment)


# In[7]:


df.isna().sum()


# In[8]:


df['Age_Label'] = None
df['Churn_Label']=None
df['Month_Label']=None


# In[9]:


df.loc[df['Age'] < 30, 'Age_Label'] = 'Under 30 years'
df.loc[df['Age'] > 65, 'Age_Label'] = 'Over 65 years'
df.loc[(df['Age']>=30) & (df['Age']<=65), 'Age_Label'] = 'From 30 to 65 years'
df.loc[df['Churn Value'] ==1, 'Churn_Label'] = 'Đã rời bỏ'
df.loc[df['Churn Value'] ==0, 'Churn_Label'] = 'Chưa rời bỏ'
df.loc[df['Tenure in Months'] <=3, 'Month_Label'] = '1 - 3 tháng'
df.loc[(df['Tenure in Months'] <=6) & (df['Tenure in Months'] >3), 'Month_Label'] = '1 - 6 tháng'
df.loc[(df['Tenure in Months'] <=12) & (df['Tenure in Months'] >6), 'Month_Label'] = '1 năm'
df.loc[(df['Tenure in Months'] <=24) & (df['Tenure in Months'] >12), 'Month_Label'] = '2 năm'
df.loc[(df['Tenure in Months'] <=60) & (df['Tenure in Months'] >24), 'Month_Label'] = '5 năm'
df.loc[(df['Tenure in Months'] >60), 'Month_Label'] = 'trên 5 năm'
df.head(5)


# In[10]:


df.info()


# In[11]:


df['Month_Label'].unique()


# In[12]:


df['Multiple Lines'].value_counts()


# ### Churn

# In[13]:


plt.figure(figsize= (10, 6))
data_pie  = df['Churn_Label'].value_counts()
labels = ['Chưa rời bỏ', 'Đã rời bỏ']
explode = [0.1, 0]
plt.pie(data_pie ,labels= labels , explode = explode , autopct="%1.2f%%", shadow= True, colors= ['#256D85', '#3BACB6'])
plt.legend()
plt.show()


# ### Giới tính

# In[14]:


gt=df.groupby(['Churn_Label', 'Gender']).count().reset_index()
fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Chưa rời bỏ', 'Đã rời bỏ'])
fig.add_trace(go.Pie(labels=gt[gt['Churn_Label']=='Chưa rời bỏ']['Gender'],
                     values=gt[gt['Churn_Label']=='Chưa rời bỏ']['Customer ID'], scalegroup='one',
                     name="Chưa rời bỏ"), 1, 1)
fig.add_trace(go.Pie(labels=gt[gt['Churn_Label']=='Đã rời bỏ']['Gender'],
                     values=gt[gt['Churn_Label']=='Đã rời bỏ']['Customer ID'], scalegroup='one',
                     name="Đã rời bỏ"), 1, 2)
fig.show()


# ### Tuổi

# In[15]:


tuoi=df.groupby(['Churn_Label','Age_Label']).count().reset_index()
fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Chưa rời bỏ', 'Đã rời bỏ'])
fig.add_trace(go.Pie(labels=tuoi[tuoi['Churn_Label']=='Chưa rời bỏ']['Age_Label'],
                     values=tuoi[tuoi['Churn_Label']=='Chưa rời bỏ']['Customer ID'], scalegroup='one',
                     name="Chưa rời bỏ"), 1, 1)
fig.add_trace(go.Pie(labels=tuoi[tuoi['Churn_Label']=='Đã rời bỏ']['Age_Label'],
                     values=tuoi[tuoi['Churn_Label']=='Đã rời bỏ']['Customer ID'], scalegroup='one',
                     name="Đã rời bỏ"), 1, 2)
fig.show()


# ### Kết hôn và số người phụ thuộc

# In[16]:


kh=df[['Churn_Label','Married','Number of Dependents','Customer ID']]
kh=kh.groupby(['Churn_Label','Married','Number of Dependents']).count().reset_index()
kh


# In[17]:


fig = make_subplots(2, 1, subplot_titles=['Chưa rời bỏ', 'Đã rời bỏ'])
fig.add_trace(go.Bar(x=kh['Number of Dependents'].unique().tolist(),
                     y =kh[kh['Churn_Label']=='Chưa rời bỏ'][kh['Married']=='Yes']['Customer ID'],
                     text=kh[kh['Churn_Label']=='Chưa rời bỏ'][kh['Married']=='Yes']['Customer ID'],
                     name="Chưa rời bỏ và đã kết hôn"),1,1)
fig.add_trace(go.Bar(x=kh['Number of Dependents'].unique().tolist(),
                     y =kh[kh['Churn_Label']=='Chưa rời bỏ'][kh['Married']=='No']['Customer ID'],
                     text=kh[kh['Churn_Label']=='Chưa rời bỏ'][kh['Married']=='No']['Customer ID'],
                     name="Chưa rời bỏ và độc thân"),1,1)
fig.add_trace(go.Bar(x=kh['Number of Dependents'].unique().tolist(),
                     y =kh[kh['Churn_Label']=='Đã rời bỏ'][kh['Married']=='Yes']['Customer ID'],
                     text=kh[kh['Churn_Label']=='Đã rời bỏ'][kh['Married']=='Yes']['Customer ID'],
                     name="Đã rời bỏ và đã kết hôn"), 2,1)
fig.add_trace(go.Bar(x=kh['Number of Dependents'].unique().tolist(),
                     y =kh[kh['Churn_Label']=='Đã rời bỏ'][kh['Married']=='No']['Customer ID'],
                     text=kh[kh['Churn_Label']=='Đã rời bỏ'][kh['Married']=='No']['Customer ID'],
                     name="Đã rời bỏ và độc thân"), 2, 1)
fig.update_layout(
    autosize=False,
    width=1000,
    height=1000
)
fig.show()


# ### City

# In[18]:


ct=df[['Churn_Label','City','Customer ID']]
ct=ct.groupby(['Churn_Label','City']).count().reset_index()
ct


# In[19]:


# import plotly.express as px
# fig = px.scatter_geo(df, locations="iso_alpha", color="continent",
#                      hover_name="country", size="pop",
#                      animation_frame="year",
#                      projection="natural earth")

# fig.show()


# In[20]:



#plot
# fig = px.choropleth(ct, locations="iso_alpha",
#                     color="Customer ID",
#                     hover_name="City",
#                     animation_frame="Churn_Label",
#                     title = "Daily new COVID cases",
#                     scope ='asia',  color_continuous_scale=px.colors.sequential.PuRd)

# fig["layout"].pop("updatemenus")
# fig.show()


# In[21]:


mon=df.groupby(['Churn_Label','Month_Label']).count().reset_index()
mon
fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Chưa rời bỏ', 'Đã rời bỏ'])
fig.add_trace(go.Pie(labels=mon[mon['Churn_Label']=='Chưa rời bỏ']['Month_Label'],
                     values=mon[mon['Churn_Label']=='Chưa rời bỏ']['Customer ID'], scalegroup='one',
                     name="Chưa rời bỏ"), 1, 1)
fig.add_trace(go.Pie(labels=mon[mon['Churn_Label']=='Đã rời bỏ']['Month_Label'],
                     values=mon[mon['Churn_Label']=='Đã rời bỏ']['Customer ID'], scalegroup='one',
                     name="Đã rời bỏ"), 1, 2)
fig.show()


# In[22]:


fig,ax = plt.subplots(3,3, figsize=(20,20))
gb = df.groupby("Churn_Label")["Multiple Lines"].value_counts().to_frame().rename({"Multiple Lines": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Multiple Lines",ax=ax[0][1]).set_title("Multiple Lines")
gb = df.groupby("Churn_Label")["Phone Service"].value_counts().to_frame().rename({"Phone Service": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Phone Service",ax=ax[0][0]).set_title("Phone Service")
gb = df.groupby("Churn_Label")["Internet Service"].value_counts().to_frame().rename({"Internet Service": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Internet Service",ax=ax[0][2]).set_title("Internet Service")
gb = df.groupby("Churn_Label")["Online Security"].value_counts().to_frame().rename({"Online Security": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Online Security",ax=ax[1][0]).set_title("Online Security")
gb = df.groupby("Churn_Label")["Online Backup"].value_counts().to_frame().rename({"Online Backup": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Online Backup",ax=ax[1][1]).set_title("Online Backup")
gb = df.groupby("Churn_Label")["Device Protection Plan"].value_counts().to_frame().rename({"Device Protection Plan": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Device Protection Plan",ax=ax[1][2]).set_title("Device Protection Plan")
gb = df.groupby("Churn_Label")["Premium Tech Support"].value_counts().to_frame().rename({"Premium Tech Support": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Premium Tech Support",ax=ax[2][0]).set_title("Premium Tech Support")
gb = df.groupby("Churn_Label")["Streaming TV"].value_counts().to_frame().rename({"Streaming TV": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Streaming TV",ax=ax[2][1]).set_title("Streaming TV")
gb = df.groupby("Churn_Label")["Streaming Movies"].value_counts().to_frame().rename({"Streaming Movies": "Number of Customers"}, axis = 1).reset_index()
sns.barplot(x = "Churn_Label", y = "Number of Customers", data = gb, hue = "Streaming Movies",ax=ax[2][2]).set_title("Streaming Movies")


# In[23]:


fig,ax = plt.subplots(1,3, figsize=(20,5))
sns.histplot(data=df[df['Contract']=='Month-to-Month'], x='Tenure in Months',kde=True,ax=ax[0]).set_title("Month-to-Month")
sns.histplot(data=df[df['Contract']=='One Year'], x='Tenure in Months',kde=True,ax=ax[1]).set_title("One Year")
sns.histplot(data=df[df['Contract']=='Two Year'], x='Tenure in Months',kde=True,ax=ax[2]).set_title("Two Year")


# In[24]:


fig,ax = plt.subplots(1,3, figsize=(20,5))
sns.histplot(data=df[df['Payment Method']=='Bank Withdrawal'], x='Tenure in Months',kde=True,ax=ax[0]).set_title("Bank Withdrawal")
sns.histplot(data=df[df['Payment Method']=='Credit Card'], x='Tenure in Months',kde=True,ax=ax[1]).set_title("Credit Card")
sns.histplot(data=df[df['Payment Method']=='Mailed Check'], x='Tenure in Months',kde=True,ax=ax[2]).set_title("Mailed Check")


# In[25]:


import plotly.express as px
fig1 = px.sunburst(df,path=['Satisfaction Score','Churn_Label'],template="plotly")
fig1.show()


# In[26]:


fig=px.pie(df,values=df["Churn Category"].value_counts()[1:],
           names=["Competitor",'Attitude','Dissatisfaction','Price','Other'],
           hole=.4,color_discrete_sequence=px.colors.qualitative.Pastel,template="plotly")
fig.update_layout(title_font_size=30)
fig.show()


# In[27]:


a=df.groupby("Churn_Label")
l=[]
for name,group in a:
    l.append(group)
notchurn=l[0].reset_index()
churn=l[1].reset_index()
notchurn.info()


# In[28]:


nc=notchurn.iloc[:,[9,10,11,12,13,14,15,16,17,18,19]]
nc


# In[29]:


dataset=[]
for i in range(len(nc)):
    x=[]
    for j in nc.columns[:]:
        if nc[j][i]=='Yes':
            x.append(j)
    dataset.append(x)
dataset


# In[30]:


get_ipython().system('pip install mlxtend')


# In[31]:


from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import association_rules, apriori


# In[32]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
nc = pd.DataFrame(te_ary, columns=te.columns_)
nc


# In[33]:


frequent_itemsets = apriori (nc, min_support = 0.05, use_colnames = True) 
rules = association_rules (frequent_itemsets, metric='lift', min_threshold=1.08) 
frequent_itemsets['length'] =frequent_itemsets['itemsets'].apply(lambda x: len(x))
rules.sort_values('confidence',ascending=False)


# In[34]:


c=churn.iloc[:,[9,10,11,12,13,14,15,16,17,18,19]]
c


# In[35]:


dataset=[]
for i in range(len(c)):
    x=[]
    for j in c.columns[:]:
        if c[j][i]=='Yes':
            x.append(j)
    dataset.append(x)
dataset


# In[36]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
c = pd.DataFrame(te_ary, columns=te.columns_)
c


# In[37]:


frequent_itemsets = apriori (c, min_support = 0.05, use_colnames = True) 
rules = association_rules (frequent_itemsets, metric='lift', min_threshold=1.08) 
frequent_itemsets['length'] =frequent_itemsets['itemsets'].apply(lambda x: len(x))
rules.sort_values('confidence',ascending=False)


# In[38]:


do_dummy_cols = ['Gender', 'Married', 'Number of Dependents', 'Phone Service', 'Multiple Lines',
       'Internet Service', 'Online Security', 'Online Backup', 'Device Protection Plan',
       'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract',
       'Paperless Billing', 'Payment Method']
model_df = df.copy()
model_df = pd.get_dummies(model_df, columns=do_dummy_cols)
model_df


# In[39]:


model_df.drop(columns=['Customer ID','City','Offer','Satisfaction Score','CLTV','Age_Label',
                       'Churn Category','Churn_Label','Month_Label'], inplace=True)
model_df.info()


# ## Churn

# In[40]:


y = model_df['Churn Value']
X = model_df.drop(columns=['Tenure in Months','Churn Value'])


# In[41]:


get_ipython().system('pip install xgboost')


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay, r2_score


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[44]:


sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# In[45]:


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


# In[46]:


xgb_clf = XGBClassifier(n_estimators=500, max_depth=1, max_leaves=2, random_state=0)
xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
clf = 'XGBoost'
model_eval(clf, y_test, y_pred)


# ## Tenure

# In[77]:


y = model_df['Tenure in Months']
X = model_df.drop(columns=['Tenure in Months','Churn Value'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc_r = StandardScaler()
X_train= sc_r.fit_transform(X_train)
X_test= sc_r.transform(X_test)
X_test[1]


# In[83]:


xgb_r = XGBRegressor(n_estimators=1000, random_state=42)
xgb_r.fit(X_train,y_train)
y_pred = xgb_r.predict(X_test)
print(f'R Squared Score: {r2_score(y_pred, y_test)}')


# ## Predictions on a Custom User Data

# In[95]:


user_input=X_test[:1]
user_input


# In[108]:


churn = ['No', 'Yes']
print('Số tháng khác hàng ở lại với cty:',xgb_r.predict(X_test)[0])
print('KH có rời bỏ không?',churn[xgb_clf.predict(user_input)[0]])
print('xác suất khách hàng ở lại công ty là:',(100-xgb_clf.predict_proba(user_input)[:, 1][0]*100))


# In[47]:


# export_csv = df.to_csv (r'Telco_customer_churn.csv', index = None, header=True)


# In[ ]:




