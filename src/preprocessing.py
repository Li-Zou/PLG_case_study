import os
import pandas as pd
import chardet
import numpy as np
import dateutil.parser
from datetime import datetime
from currency_converter import CurrencyConverter
from scipy import stats

#Check the encodings for file advanced_case_study_data.csv
with open('./data/raw/advanced_case_study_data.csv', 'rb') as f:
    data = f.read(200000)
result = chardet.detect(data)

#load data
df = pd.read_csv('./data/raw/advanced_case_study_data.csv', encoding=result['encoding'])
#print(df.head())

#some columns contain a mix of uppercase and lowercase letters; convert them all to lowercase
columns=df.columns
#col1=['legacy_system_id','marketing_channel','payment_method', 'engagement_history', 
      #'campaign_cohort','treatment_sent_flag', 'baseline_churn_risk_band','churned']
for i in columns:
    try:
        df[i]=df[i].str.lower()
    except:
        pass
    
#handling missing value
df.replace(['unknown','missing'], np.nan, inplace=True)
#check the missing value for each feature
missing_count=df.isna().sum()
missing_count=round(missing_count[missing_count!=0]/df.shape[0],2)

#drop any features whose proportion of missing values exceeds a specified threshold (e.g., 0.2)
missing_count.sort_values(inplace=True,ascending=False)
missing_count=missing_count[missing_count>=0.2]
feature_remove=list(missing_count.index)
df1=df.drop(columns=feature_remove)

val1=df1.isnull().values.any(axis=1).sum()/df1.shape[0]
#print(f"Proportion of samples that contain at least one missing value.: {round(100*val1,2)}%")
df1=df1.dropna()##remove samples that contain at least one missing value

######handle date formats
###Convert different date formats to a given date format
df2=df1.copy()
df2['subscription_date']=df2['subscription_date'].apply(lambda x: dateutil.parser.parse(x).strftime("%Y-%m-%d"))
df2['observation_end_date']=df2['observation_end_date'].apply(lambda x: dateutil.parser.parse(x).strftime("%Y-%m-%d"))
#these two date formats may not be usable directly, so a new feature (subscription_month_length) is created to make use of this information
o1=df2['subscription_date']+','+df2['observation_end_date']
def month_difference(date_str):
    a=date_str.split(',')
    d1 = datetime.strptime(a[1], '%Y-%m-%d')
    d2 = datetime.strptime(a[0], '%Y-%m-%d')
    return d1.month - d2.month + 12*(d1.year - d2.year)
df2['subscription_month_length']=o1.apply(lambda x: month_difference(x))
    
#replace "," with "."
df2['add_ons']=df2['add_ons'].apply(lambda x: int(float((x.replace(',','.')))))

#convert all currency values to euros
c = CurrencyConverter()
def Currency_Conversion(x:str) ->float:
    x=x.replace(',','.')
    if x.startswith("$"):  
        return round(c.convert(float(x[1:]), 'EUR', 'USD'),2)
    elif x.startswith("€"):
        return float(x[1:])
    else:
        return float(x)
for variable in ['monthly_spend_estimated','offer_cost_eur','historic_revenue_12m','revenue_next_12m_observed']:
    df2[variable]=df2[variable].apply(lambda x: Currency_Conversion(x))
 
#transform the two formats into a single one for the donation_share_charity feature
def handel_donation_share_charity(x:str) ->float:
    x=x.replace(',','.')
    if float(x)<1:  
        return round(float(x)*100,2)
    else:
        return round(float(x),2)
df2['donation_share_charity']=df2['donation_share_charity'].apply(lambda x: handel_donation_share_charity(x))

#remove the '+' from the web_sessions_90d_raw feature; to be honest, I don't fully understand what the '+' indicates
df2['web_sessions_90d_raw']=df2['web_sessions_90d_raw'].apply(lambda x: x.strip('+'))

##'service_contacts_12m' column contain 366 outliers, these samples corresponding to these outliers needed to be removed
df2=df2[(np.abs(stats.zscore(df2['service_contacts_12m'])) < 3)]

df2['treatment_sent_flag']=df2['treatment_sent_flag'].map({'true': 1, 'y': 1,'yes':1,'1':1, 'false': 0, 'n': 0,'no':0,'0':0})

#print(df2['baseline_churn_risk_band'].unique())
df2['baseline_churn_risk_band']=df2['baseline_churn_risk_band'].str.replace("_",' ')

df2['churned']=df2['churned'].map({'true': 1, 'y': 1,'yes':1,'1':1, 'false': 0, 'n': 0,'no':0,'0':0})


columns=list(df2.columns)
di={}
for i in columns:
    di[i]=df2[i].dtype
#print(di)

col1=['participant_age','web_sessions_90d_raw']
for variable in col1:
    df2[variable]=df2[variable].apply(lambda x: int(x))

df2.to_csv(r'F:\PLG\data\processed\clean_data_ver0.csv',index=False)  

'''only consider 3 situations: 
    'campaign_cohort'='Control' and 'treatment_sent_flag'=0;
    'campaign_cohort'='Variant_A' and 'treatment_sent_flag'=1;
    'campaign_cohort'='Variant_B' and 'treatment_sent_flag'=1;
'''
df2=df2.loc[((df2['campaign_cohort']=='control') & (df2['treatment_sent_flag']==0))|
           ((df2['campaign_cohort']=='variant_a') & (df2['treatment_sent_flag']==1))|
           ((df2['campaign_cohort']=='variant_b') & (df2['treatment_sent_flag']==1))]
df2.drop(columns=['legacy_system_id','subscription_date','postcode_area','observation_end_date'],inplace=True)

# store the data inside data/processed
data_path = os.path.join("data","processed")
try:
    os.makedirs(data_path)
except: pass
df2.to_csv(os.path.join(data_path,"retention_model_data.csv"))



