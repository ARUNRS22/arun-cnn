#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,train_test_split,cross_validate
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import missingno as msno
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import scipy.stats as st
from sklearn.pipeline import make_pipeline 


# In[2]:


df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\diabetes_binary_health_indicators_BRFSS2015.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


len(df[df.duplicated()])


# In[6]:


df.drop_duplicates(inplace=True)


# In[7]:


df.shape


# In[8]:


msno.bar(df)


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


sns.countplot(df['Diabetes_binary'])


# # * People who are non diabetic are more
# * This represents a class imbalance in the data

# In[12]:


df.plot(kind='kde',subplots=True,layout=(3,8),sharex=False,figsize=(15,10))


# In[13]:


df.skew().sort_values()


# # * Only 7 features are of near normal distribution
# * 4 features are right skewed
# * The rest are left skewed

# In[14]:


plt.figure(figsize=(9,7))
sns.heatmap(df.corr()[(df.corr()>0.5)|(df.corr()<-0.5)],annot=True)


# # * Medium correlation between PhysHlth and GenHlth
# * No multicollinearity is present

# In[15]:


for i in df.columns:
    print(df[i].value_counts())
    print('*************************************')


# In[16]:


for i in df.columns:
    print(df[i].value_counts()/len(df)*100)
    print('*************************************')


# In[17]:


X=df.drop('Diabetes_binary',axis=1)
Y=df['Diabetes_binary']


# In[18]:


##Finding significance of features using p values**


# In[19]:


from sklearn.feature_selection import SelectKBest,f_classif
skb=SelectKBest(f_classif,k='all')
skb.fit(X,Y)
skb.pvalues_


# In[20]:


pvals=pd.DataFrame({'Features':X.columns,'P- values':skb.pvalues_})


# In[21]:


pvals['P- values']=round(pvals['P- values'],5)


# In[22]:


pvals


# # * All the columns have p values which are near **Zero**

# In[23]:


##Visualization##


# In[24]:


plt.figure(figsize=(25,15))

for i, variable in enumerate(df):
                     plt.subplot(5,5,i+1)
                     sns.countplot(df[variable])
                     plt.tight_layout()
                     plt.title(variable)

plt.show()


# In[25]:


sns.boxplot(df['BMI'])


# In[26]:


q1=df.quantile(0.25)
q3=df.quantile(0.75)
iqr=q3-q1
df=df[~((df<(q1-1.5*iqr))|(df>(q3+1.5*iqr))).any(axis=1)]
df=df.reset_index(drop=True)


# In[27]:


df.shape


# In[28]:


sns.boxplot(df['BMI'])


# In[29]:


plt.hist(df['BMI'])
plt.show()


# In[30]:


##Splitting##


# In[31]:


X.dtypes


# In[32]:


xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=9,stratify=Y)


# In[33]:


xtrain


# In[36]:


LR=LogisticRegression()
model_lr=LR.fit(xtrain,ytrain)
ytrain_pred_lr= model_lr.predict(xtrain)
ytest_pred_lr=model_lr.predict(xtest)


# In[37]:


odds=pd.DataFrame(np.exp(model_lr.coef_),columns=X.columns)
odds=odds.T
odds.sort_values(by=0,ascending=False)


# In[38]:


confusion_matrix(ytrain,ytrain_pred_lr)


# In[39]:


confusion_matrix(ytest,ytest_pred_lr)


# In[40]:


print(classification_report(ytrain,ytrain_pred_lr))
print('********************************************************')
print(classification_report(ytest,ytest_pred_lr))


# In[41]:


fpr, tpr, thresholds = roc_curve(ytest, ytest_pred_lr)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest, ytest_pred_lr),4)))

plt.grid(True)
plt.show()


# In[42]:


decision_tree=DecisionTreeClassifier(random_state=9)
model_dt=decision_tree.fit(xtrain,ytrain)
y_pred_dt=model_dt.predict(xtest)


# In[43]:


confusion_matrix(ytest,y_pred_dt)


# In[44]:


print(classification_report(ytest,y_pred_dt))


# In[45]:


important_features = pd.DataFrame({'Features': xtrain.columns, 
                                   'Importance': model_dt.feature_importances_})
important_features.sort_values(by='Importance', ascending=False)


# In[46]:


fpr, tpr, thresholds = roc_curve(ytest, y_pred_dt)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest, y_pred_dt),4)))

plt.grid(True)
plt.show()


# In[47]:


tuned_paramaters = [{'criterion': ['entropy','gini'],
                     'max_depth': range(2,7),
                     'min_samples_split': range(2,5)
                     }]
decision_tree_classification = DecisionTreeClassifier(random_state = 10)

rf_grid = GridSearchCV(estimator = decision_tree_classification, 
                       param_grid = tuned_paramaters, 
                       cv = 5)
rf_grid_model = rf_grid.fit(xtrain, ytrain)
# get the best parameters
print('Best parameters for random forest classifier: ', rf_grid_model.best_params_, '\n')


# In[48]:


decision_tree_tuned_model = DecisionTreeClassifier(criterion = rf_grid_model.best_params_['criterion'], 
                                                   max_depth = rf_grid_model.best_params_['max_depth'], 
                                                   min_samples_split = rf_grid_model.best_params_['min_samples_split'])
scores = cross_val_score(estimator = decision_tree_tuned_model, 
                         X = xtrain, 
                         y = ytrain, 
                         cv = 5, 
                         scoring = 'roc_auc')
print("Mean ROC-AUC score after 10 fold cross validation: ", scores.mean())


# In[49]:


decision_tree_tuned_model = DecisionTreeClassifier(criterion = rf_grid_model.best_params_['criterion'], 
                                                   max_depth = rf_grid_model.best_params_['max_depth'], 
                                                   min_samples_split = rf_grid_model.best_params_['min_samples_split'])
scores = cross_validate(estimator = decision_tree_tuned_model, 
                         X = xtrain, 
                         y = ytrain, 
                         cv = 5, 
                         scoring =['accuracy','f1','precision'])
score=pd.DataFrame(scores)
score


# In[50]:


decision_tree_tuned_model = DecisionTreeClassifier(criterion='gini', 
                                                   max_depth=5, 
                                                   min_samples_split=2,random_state=10,class_weight='balanced')
model_dtt=decision_tree_tuned_model.fit(xtrain,ytrain)
y_pred_dtt=model_dtt.predict(xtest)


# In[51]:


confusion_matrix(ytest,y_pred_dtt)


# In[52]:


print(classification_report(ytest,y_pred_dtt))


# In[53]:


fpr, tpr, thresholds = roc_curve(ytest, y_pred_dtt)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest, y_pred_dtt),4)))

plt.grid(True)
plt.show()


# In[54]:


cm=confusion_matrix(ytest,y_pred_dtt)
total=cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1]
correct_classified_percentage=((cm[0][0]+cm[1][1])/total)*100
print('% of correctly classified observation:',round(correct_classified_percentage,2))
misclassified_percentage=((cm[0][1]+cm[1][0])/total)*100
print('% of misclassified observation:',round(misclassified_percentage,2))


# In[55]:


random_forest=RandomForestClassifier(random_state=10,class_weight='balanced')
model_rf=random_forest.fit(xtrain,ytrain)


# In[56]:


ytrain_pred_rf=model_rf.predict(xtrain)
ytest_pred_rf=model_rf.predict(xtest)


# In[57]:


print(classification_report(ytrain,ytrain_pred_rf))
print('--------------------------------------------------------------')
print(classification_report(ytest,ytest_pred_rf))


# # * The null model (Random Forest) is overfitting

# In[58]:


fpr, tpr, thresholds = roc_curve(ytest, ytest_pred_rf)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest, ytest_pred_rf),4)))

plt.grid(True)
plt.show()


# In[59]:


tuned_params=[{'n_estimators':[75,100],
              'max_depth': [5,7],
               'min_samples_split': [2,5],
               'criterion' : ['entropy', 'gini']}]
tree_grid=GridSearchCV(estimator=RandomForestClassifier(),param_grid=tuned_params,cv=5)
model=tree_grid.fit(xtrain,ytrain)
model.best_params_


# In[60]:


random_forest_tuned=RandomForestClassifier(criterion='gini',max_depth=7,min_samples_split=2,
                                           n_estimators=75,class_weight='balanced')
model_rft=random_forest_tuned.fit(xtrain,ytrain)


# In[61]:


ytrain_pred_rft=model_rft.predict(xtrain)
ytest_pred_rft=model_rft.predict(xtest)


# In[62]:


confusion_matrix(ytest,ytest_pred_rft)


# In[63]:


print(classification_report(ytrain,ytrain_pred_rft))
print('--------------------------------------------------------------')
print(classification_report(ytest,ytest_pred_rft))


# In[64]:


fpr, tpr, thresholds = roc_curve(ytest, ytest_pred_rft)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest, ytest_pred_rft),4)))

plt.grid(True)
plt.show()


# In[65]:


cm=confusion_matrix(ytest,ytest_pred_rft)
total=cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1]
correct_classified_percentage=((cm[0][0]+cm[1][1])/total)*100
print('% of correctly classified observation:',round(correct_classified_percentage,2))
misclassified_percentage=((cm[0][1]+cm[1][0])/total)*100
print('% of misclassified observation:',round(misclassified_percentage,2))


# In[66]:


important_features = pd.DataFrame({'Features': xtrain.columns, 
                                   'Importance': model_rf.feature_importances_})

important_features = important_features.sort_values('Importance', ascending = False)

sns.barplot(x = 'Importance', y = 'Features', data = important_features)
plt.title('Feature Importance', fontsize = 15)
plt.xlabel('Importance', fontsize = 15)
plt.ylabel('Features', fontsize = 15)

plt.show()


# In[67]:


gnb=GaussianNB()
model_gnb=gnb.fit(xtrain,ytrain)
y_pred_gnb=model_gnb.predict(xtest)
print(classification_report(ytest,y_pred_gnb))


# In[68]:


confusion_matrix(ytest,y_pred_gnb)


# In[69]:


fpr, tpr, thresholds = roc_curve(ytest, y_pred_gnb)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest, y_pred_gnb),4)))

plt.grid(True)
plt.show()


# In[70]:


X.columns


# In[71]:


x_new=X.drop(['BMI','MentHlth','PhysHlth'],axis=1)
SS=StandardScaler()
x_ss=pd.DataFrame(SS.fit_transform(X[['BMI','MentHlth','PhysHlth']]),columns=['BMI','MentHlth','PhysHlth'])


# In[72]:


x_s=pd.concat([x_new.reset_index(drop=True),x_ss.reset_index(drop=True)],axis=1)


# In[73]:


x_s.shape


# In[74]:


xtrain_,xtest_,ytrain_,ytest_=train_test_split(x_s,Y,test_size=0.3,random_state=9,stratify=Y)


# In[75]:


ada=AdaBoostClassifier(random_state=9)


# In[76]:


model_ada=ada.fit(xtrain_,ytrain_)


# In[77]:


y_pred_ada=model_ada.predict(xtest_)


# In[78]:


confusion_matrix(ytest_,y_pred_ada)


# In[79]:


print(classification_report(ytest_,y_pred_ada))


# In[80]:


fpr, tpr, thresholds = roc_curve(ytest, y_pred_ada)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest, y_pred_ada),4)))

plt.grid(True)
plt.show()


# In[81]:


ada = AdaBoostClassifier()
params = {'n_estimators' : [50,75,100],
          'learning_rate' :  [0.1, 0.05]}
ada_grid = GridSearchCV(estimator=ada, param_grid=params, cv = 5)
ada_model = ada_grid.fit(xtrain_, ytrain_)
print(ada_model.best_params_)


# In[82]:


ada_tuned=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,learning_rate=0.1,random_state=9)
model_adat=ada_tuned.fit(xtrain_,ytrain_)


# In[83]:


y_pred_adat=model_adat.predict(xtest)


# In[84]:


confusion_matrix(ytest_,y_pred_adat)


# In[85]:


print(classification_report(ytest_,y_pred_adat))


# In[86]:


fpr, tpr, thresholds = roc_curve(ytest_, y_pred_adat)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest_, y_pred_adat),4)))

plt.grid(True)
plt.show()


# In[87]:


grad = GradientBoostingClassifier()
model_grad = grad.fit(xtrain_, ytrain_)
y_pred_grad =model_grad.predict(xtest_)

print(metrics.accuracy_score(ytest_, y_pred_grad))


# In[88]:


confusion_matrix(ytest_,y_pred_grad)


# In[89]:


print(classification_report(ytest_,y_pred_grad))


# In[90]:


fpr, tpr, thresholds = roc_curve(ytest_, y_pred_grad)
plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 12)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 12)

plt.text(x = 0.6, y = 0.2, s = ('AUC Score:',round(metrics.roc_auc_score(ytest_, y_pred_grad),4)))

plt.grid(True)
plt.show()


# In[91]:


grad = GradientBoostingClassifier()
params = {'n_estimators' : [100, 150],
          'learning_rate' :  [0.1, 0.05]}
grad_grid = GridSearchCV(estimator=grad, param_grid=params, cv = 5)
grad_model = grad_grid.fit(xtrain_, ytrain_)
print(grad_model.best_params_)


# In[92]:


grad_tuned = GradientBoostingClassifier(n_estimators=150,learning_rate=0.1,random_state=9)
model_gradt = grad_tuned.fit(xtrain_, ytrain_)
y_pred_gradt =model_gradt.predict(xtest_)

print(metrics.accuracy_score(ytest_, y_pred_gradt))


# In[93]:


confusion_matrix(ytest_,y_pred_gradt)


# In[94]:


print(classification_report(ytest_,y_pred_gradt))


# In[95]:


xgb=XGBClassifier()

params = {      "n_estimators": st.randint(3, 40),
                "max_depth": st.randint(3, 40),
                "learning_rate": st.uniform(0.05, 0.4),
                "gamma": st.uniform(0, 10),
                "min_child_weight": st.expon(0, 50)}
model_xgb= RandomizedSearchCV(xgb, params, cv=5,
                             n_jobs=1, n_iter=100) 

model_xgb.fit(xtrain_, ytrain_)  
 
y_pred_xgb= model_xgb.predict(xtest_)


# In[96]:


model_xgb.feature_names_in_


# In[174]:


xtrain_.columns


# In[89]:


model_xgb.best_params_


# In[90]:


confusion_matrix(ytest_, y_pred_xgb)


# In[91]:


print(classification_report(ytest_,y_pred_xgb))


# In[92]:


xgb_tuned=XGBClassifier(gamma= 6,
                         learning_rate= 0.2,
                         max_depth= 5,
                         n_estimators= 22)


# In[93]:


model_xgbt=xgb_tuned.fit(xtrain_,ytrain_)
y_pred_xgbt=model_xgbt.predict(xtest_)


# In[94]:


confusion_matrix(ytest_,y_pred_xgbt)


# In[95]:


print(classification_report(ytest_,y_pred_xgbt))


# In[96]:


pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'gnb':make_pipeline(StandardScaler(), GaussianNB()),
    'dtc':make_pipeline(StandardScaler(),DecisionTreeClassifier()),
    'xg':make_pipeline(StandardScaler(),XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_depth=7,
              min_child_weight=6,
              n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=1))
}


# In[97]:


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(xtrain_, ytrain_)
    fit_models[algo] = model


# In[98]:


from sklearn.metrics import accuracy_score
al=[]
ac=[]
for algo, model in fit_models.items():
    y_pred_pipe = model.predict(xtest_)
    al.append(algo)
    ac.append(accuracy_score(ytest_, y_pred_pipe))
    print(algo, accuracy_score(ytest_, y_pred_pipe))


# In[139]:


fit_models["lr"].predict(xtest_)


# In[102]:


algo


# In[104]:


fit_models.items


# In[141]:


model


# In[103]:


ac


# In[99]:


al


# In[100]:


ac


# In[101]:


XGBClassifier


# In[2]:


pip install -U flask-cors


# In[149]:


import pickle
from sklearn.linear_model import LinearRegression

lr_obj=model


with open(r"C:\Users\User\OneDrive\Desktop\PGA 23\Project diabetsmodel.pickle","wb") as s:
    pickle.dump(lr_obj,s)


# In[150]:


with open(r"C:\Users\User\OneDrive\Desktop\PGA 23\Project diabetsmodel.pickle","rb") as s:
    ee=pickle.load(s)


# In[151]:


ee


# In[169]:


def prediction(x):
    with open(r"C:\Users\User\OneDrive\Desktop\PGA 23\Project diabetsmodel.pickle","rb") as s:
        model=pickle.load(s)
    return model.predict(x)


# In[170]:


prediction(xtest)


# In[161]:


ee.predict(xtest_)


# In[148]:


ee(xtest_)


# In[176]:


df["Age"]


# In[178]:


df.columns.str=="0"


# In[179]:


df.columns


# In[ ]:




