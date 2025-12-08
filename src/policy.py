import pandas as pd
import os

df=pd.read_csv('./data/outputs/individual_treatment_effects.csv')

#churn_rate=(df['churned']==1).sum()/len(df)
#revenue_per_user=df['monthly_spend_estimated']*12*(1-0.01*df['donation_share_charity'])#-df['offer_cost_eur']
#average_revenue_per_user=revenue_per_user.mean()
#LTV_past_year=(average_revenue_per_user/churn_rate).item()

df=df.sort_values(by=['ITE_variant_a','ITE_variant_b'],ascending=True)

N=df.shape[0]
N_target=int(N*0.2)

df1=df.iloc[:N_target,:]
df1=df1.assign(variant_a_or_b='a')
#df1['variant_a_or_b']='a'
df1.loc[(df1['ITE_variant_a']>-0.3) & (df1['ITE_variant_a']>df1['ITE_variant_b']),'variant_a_or_b']='b'

total_cost=(df1['variant_a_or_b']=='a').sum()*10+(df1['variant_a_or_b']=='b').sum()*20
print(f"an approximate for the value of saving a subscription: {total_cost/N_target}")
#revenue_all=df['monthly_spend_estimated']*12*(1-0.01*df['donation_share_charity'])

data_path = os.path.join("data","outputs")
try:
    os.makedirs(data_path)
except: pass
df1.to_csv(os.path.join(data_path,"targeting_customer.csv"))     
    


