import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
import os

def plot_count_pie(df,variable):
    import matplotlib.pyplot as plt
    import seaborn as sns
    variable='campaign_cohort'
    plt.figure(figsize=(6,4))
    plt.subplot(1,2,1)
    sns.countplot(x=variable,data=df)
    plt.title(f'Count Plot - {variable}')
    plt.subplot(1,2,2)
    counts=df[variable].value_counts()
    plt.pie(counts,labels=counts.index,autopct='%0.2f%%')
    plt.title(f'Pie Plot - {variable}')
    plt.tight_layout()
    plt.show()
#plot_count_pie(df,variable='campaign_cohort')
#plot_count_pie(df,variable='churned')
def two_proportion_z_test(df1, variable,alpha):
    df1_A=df1.loc[(df1['campaign_cohort']=='control') | (df1['campaign_cohort']==variable)]

    N_con=(df1_A["campaign_cohort"] == "control").sum().item()
    N_exp =(df1_A["campaign_cohort"] == variable).sum().item()
    
    # calculating the total number of churn per group by summing 1's
    X_con = df1_A.groupby("campaign_cohort")["churned"].sum().loc["control"].item()
    X_exp =df1_A.groupby("campaign_cohort")["churned"].sum().loc[variable].item()
    
    # computing the estimate of churn probability per group
    p_con_hat = X_con/N_con
    p_exp_hat = X_exp/N_exp
    
    # computing the estimate of pooled churned probability
    p_pooled_hat = (X_con+X_exp)/(N_con + N_exp)
    # computing the estimate of pooled variance
    pooled_variance = p_pooled_hat * (1-p_pooled_hat) * (1/N_con + 1/N_exp)
    # computing the standard error of the test
    SE = np.sqrt(pooled_variance)
    
    # computing the test statistics of Z-test
    Test_stat = round((p_exp_hat - p_con_hat)/SE,3)
    #ATE estimation (difference in means)
    ATE=round((p_exp_hat - p_con_hat) ,3)
    # critical value of the Z-test
    Z_crit = round(norm.ppf(1-alpha/2),3)
    #calculating p value
    p_value = round(2 * norm.sf(abs(Test_stat)).item(),3)#sf--survival function 
    # Calculate the Confidence Interval (CI) for a 2-sample Z-test
    ## Calculate the lower and upper bounds of the confidence interval
    CI = [round((p_exp_hat - p_con_hat) - SE*Z_crit, 3).item(),  
        round((p_exp_hat - p_con_hat) + SE*Z_crit, 3) .item()  ]
    return [ATE, CI, p_value, Test_stat]
def AB_test(df):
    #alpha: significance level
    ATE_a=two_proportion_z_test(df, variable='variant_a',alpha=0.05)
    ATE_b=two_proportion_z_test(df, variable='variant_b',alpha=0.05)
    
    df_age_low=df.loc[df['participant_age']<=50]
    df_age_high=df.loc[df['participant_age']>50]
    ATE_a_age_low=two_proportion_z_test(df_age_low, variable='variant_a', alpha=0.05)
    ATE_a_age_high=two_proportion_z_test(df_age_high, variable='variant_a', alpha=0.05)
    
    ATE_b_age_low=two_proportion_z_test(df_age_low, variable='variant_b', alpha=0.05)
    ATE_b_age_high=two_proportion_z_test(df_age_high, variable='variant_b', alpha=0.05)
    
    d1=pd.DataFrame({'variant_a':ATE_a,'variant_b':ATE_b,
                                'Age<=50 (variant_a)':ATE_a_age_low, 'Age>50 (variant_a)':ATE_a_age_high,
                                'Age<=50 (variant_b)':ATE_b_age_low, 'Age>50 (variant_b)':ATE_b_age_high},
                                index=['ATE', 'confidence_intervals', 'p_value','Test_stat'])
    d2={}
    col1=list(df['baseline_churn_risk_band'].unique())
    for i in col1:
        df_seg=df.loc[df['baseline_churn_risk_band']==i]
        ATE_seg_a=two_proportion_z_test(df_seg, variable='variant_a', alpha=0.05)
        ATE_seg_b=two_proportion_z_test(df_age_high, variable='variant_b', alpha=0.05)
        d2[i+' (variant_a)']=ATE_seg_a
        d2[i+' (variant_b)']=ATE_seg_b
    d2=pd.DataFrame(d2,index=['ATE', 'confidence_intervals', 'p_value','Test_stat'])
    AB_test_result=pd.merge(d1,d2,left_index=True, right_index=True)
    
    data_path = os.path.join("data","outputs")
    try:
        os.makedirs(data_path)
    except: pass
    AB_test_result.to_csv(os.path.join(data_path,"AB_test_result.csv")) 
def get_ITE_prepration(df,variable='variant_a'):
    ##Remove features that are not useful to train ML models
    df1=df.drop(columns=['customer_id','subscription_id','treatment_sent_flag','revenue_next_12m_observed'])
    #label encodling for ordinal data
    df1['baseline_churn_risk_band']=df1['baseline_churn_risk_band'].map({'low':0,'medium':1,'high':2,'very high':3})
    ##One hot encoding for nominal data
    df1 = pd.get_dummies(data=df1, columns=['marketing_channel','country_code'],dtype=float, drop_first=True)   
    
    df_exp=df1.loc[df1['campaign_cohort']==variable]
    df_con=df1.loc[df1['campaign_cohort']=='control']
    
    #for model 1
    X1 = df_exp.drop(columns=['churned','campaign_cohort']).values
    y1 = df_exp['churned'].values#.astype(np.float32)
    clf1=RandomForestClassifier(n_estimators=100,random_state=1)
    clf1.fit(X1, y1)
    
    #for model 2
    X2 = df_con.drop(columns=['churned','campaign_cohort']).values
    y2 = df_con['churned'].values#.astype(np.float32)
    clf2=RandomForestClassifier(n_estimators=100,random_state=1)
    clf2.fit(X2, y2)
    
    
    X_all = df1.drop(columns=['churned','campaign_cohort']).values
    y_pred_prob1 = clf1.predict_proba(X_all)[:, 1]
    y_pred_prob2 = clf2.predict_proba(X_all)[:, 1]
    #ITE=y_pred_prob1-y_pred_prob2
    return y_pred_prob1-y_pred_prob2
def get_ITE(df):
    ITE_a=get_ITE_prepration(df,variable='variant_a')
    ITE_b=get_ITE_prepration(df,variable='variant_b')
    df['ITE_variant_a']=ITE_a
    df['ITE_variant_b']=ITE_b
    
    df1=df.sort_values(by=['ITE_variant_a'],ascending=True)
    
    data_path = os.path.join("data","outputs")
    try:
        os.makedirs(data_path)
    except: pass
    df1.to_csv(os.path.join(data_path,"individual_treatment_effects.csv"),index=False) 
    #print(df1[['ITE_variant_a','ITE_variant_b']].corr())
df=pd.read_csv('./data/processed/retention_model_data.csv')
AB_test(df) 
get_ITE(df)




