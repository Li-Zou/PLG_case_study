import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

df=pd.read_csv('./data/processed/retention_model_data.csv')

##Remove features that are not useful to train ML models
df.drop(columns=['customer_id','subscription_id','campaign_cohort','treatment_sent_flag','revenue_next_12m_observed'],inplace=True)

#label encodling for ordinal data
df['baseline_churn_risk_band']=df['baseline_churn_risk_band'].map({'low':0,'medium':1,'high':2,'very high':3})

##One hot encoding for nominal data
df1 = pd.get_dummies(data=df, columns=['marketing_channel','country_code'],dtype=float, drop_first=True)
X = df1.drop('churned',axis='columns').values
y = df1['churned'].values#.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/5), random_state=1, stratify=y)

##handle imbalanced data by oversampling
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)
#print((y_train==0).sum()/len(y_train))#0.5

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def all_eval_metrics(y_test,y_pred_prob):
    acc=[]
    acc.append(roc_auc_score(y_test, y_pred_prob))
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    acc.append(auc(recall,precision))
    return acc

def KNN_my(X_train, X_test, y_train, y_test):
    param_grid={'n_neighbors': np.arange(2, 10, 2)}
    knn=KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, refit = True, verbose = 3)##the default 5-fold cross validation
    grid.fit(X_train, y_train) ## fitting the model for grid search 
    y_pred_prob = grid.predict_proba(X_test)[:, 1]
    ac=all_eval_metrics(y_test,y_pred_prob)
    return ac,grid

def LogisticRegression_my(X_train, X_test, y_train, y_test):
    param_grid = {   'C': [1,0.1,0.01,0.001,0.0001]}   
    grid = GridSearchCV(LogisticRegression(random_state=0), param_grid, refit = True, verbose = 3)##the default 5-fold cross validation
    grid.fit(X_train, y_train) ## fitting the model for grid search 
    y_pred_prob = grid.predict_proba(X_test)[:, 1]
    ac=all_eval_metrics(y_test,y_pred_prob)
    return ac,grid

def RandomForestClassifier_my(X_train, X_test, y_train, y_test):
    param_grid= { 'n_estimators': [100,200]} #many parameters have been tested
    grid= GridSearchCV(RandomForestClassifier(random_state=1), param_grid,refit = True, verbose = 3)
    grid.fit(X_train, y_train)
    y_pred_prob = grid.predict_proba(X_test)[:, 1]
    ac=all_eval_metrics(y_test,y_pred_prob)
    return ac,grid

def GradientBoostingClassifier_my(X_train, X_test, y_train, y_test):
    param_grid = {'n_estimators': [100,200]}  
    grid = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid, refit = True, verbose = 3)##the default 5-fold cross validation
    grid.fit(X_train, y_train) 
    y_pred_prob = grid.predict_proba(X_test)[:, 1]
    ac=all_eval_metrics(y_test,y_pred_prob)
    return ac,grid
ac1,knn_model=KNN_my(X_train, X_test, y_train, y_test)
ac2,LR_model=LogisticRegression_my(X_train, X_test, y_train, y_test)
ac3,RF_model=RandomForestClassifier_my(X_train, X_test, y_train, y_test)
ac4,GBoot_model=GradientBoostingClassifier_my(X_train, X_test, y_train, y_test)

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

prediction_acc=pd.DataFrame({'K-Nearest Neighbors':ac1,'Logistic Regression':ac2,'Random Forest':ac3,'Gradient Boosting':ac4},index=['ROC AUC','PR AUC'])

for col in prediction_acc.columns:
    prediction_acc[col]=prediction_acc[col].apply(lambda x: round(x,4))
#prediction_acc.to_csv(r'F:\PLG\data\model_save\prediction_acc.csv')  

feature_importance1=RF_model.best_estimator_.feature_importances_
feature_importance2=GBoot_model.best_estimator_.feature_importances_

g1=list(df1.drop('churned',axis='columns').columns)
g1=g1[:14]+['marketing_channel','country_code']

p1=feature_importance1[:14]
p1=np.append(p1,[sum(feature_importance1[14:20]),sum(feature_importance1[20:])])

p2=feature_importance2[:14]
p2=np.append(p2,[sum(feature_importance2[14:20]),sum(feature_importance2[20:])])

o1=pd.DataFrame({'features':g1,'RandomForest':p1,'GradientBoost':p2})

#print(o1.drop('features',axis='columns').corr())        
o1.sort_values(by=['GradientBoost'],inplace=True,ascending=False)

# store the data inside data/outputs
data_path = os.path.join("data","outputs")
try:
    os.makedirs(data_path)
except: pass
o1.to_csv(os.path.join(data_path,"feature_importance.csv"))
prediction_acc.to_csv(os.path.join(data_path,"prediction_acc.csv"))

save_model(knn_model, os.path.join(data_path,"knn_model.pkl"))
save_model(LR_model, os.path.join(data_path,"LR_model.pkl"))
save_model(RF_model, os.path.join(data_path,"RF_model.pkl"))
save_model(GBoot_model, os.path.join(data_path,"GBoost_model.pkl"))


#import seaborn as sns
#import matplotlib.pyplot as plt
#plt.figure(figsize=(10, 6))
#sns.barplot(o1,y='features',x='GradientBoost')
#plt.ylabel("feature name")
#plt.xlabel('Feature importance from GradientBoosting')
#plt.show()

#o1.sort_values(by=['RandomForest'],inplace=True,ascending=False)
#plt.figure(figsize=(10, 6))
#sns.barplot(o1,y='features',x='RandomForest')
#plt.ylabel("feature name")
#plt.xlabel('Feature importance from RandomForest')
#plt.show()

def LTV():
    df=pd.read_csv('./data/processed/retention_model_data.csv')
    ## Lifetime Value=(Average Yearly Revenue per User) × (Gross Margin %) × (Expected Lifetime in Years)
    churn_rate=(df['churned']==1).sum()/len(df)
    revenue_per_user=df['monthly_spend_estimated']*12*(1-0.01*df['donation_share_charity'])#-df['offer_cost_eur']
    average_revenue_per_user=revenue_per_user.mean()
    
    LTV_past_year=(average_revenue_per_user/churn_rate).item()
    LTV_next_year=((df['revenue_next_12m_observed'].mean())/churn_rate).item()
    LTV_mean=(LTV_past_year+LTV_next_year)/2
    
    LTV=pd.DataFrame({'LTV':[round(LTV_past_year,2),round(LTV_next_year,2),round(LTV_mean,2)]},
                    index=['LTV_past_year','LTV_next_year','LTV_mean'])
    
    data_path = os.path.join("data","outputs")
    try:
        os.makedirs(data_path)
    except: pass
    LTV.to_csv(os.path.join(data_path,"LTV.csv"))   
LTV()

    






