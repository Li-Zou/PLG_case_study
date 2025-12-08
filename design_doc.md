# Data Scientist Take-Home Assignment

## Part A — Data Engineering & Pipeline
	(`src/preprocessing.py`)
	## 1. Exploratory Data Analysis (EDA)
	## 2. Cleaning & Standardization
	## 3. Feature Engineering
	## 4. Reproducibility
1.1.  Check the raw file’s encoding and load it using the corresponding encoding.
1.2.  Some columns contain a mix of uppercase and lowercase letters; convert them all to lowercase
1.3.  Remove missing values
	  Replace ['unknown','missing'] with np.nan
	  First, check the proposition of missing value of each feature,
	  and drop any features whose proportion of missing values exceeds a specified threshold (e.g., 0.2).
	  Second, drop samples that contain at least one missing value.
1.4.  Convert different date formats to a given date format, e.g,. '%Y-%m-%d'.
	  Create a new feature (subscription_month_length)to make use of this date information.
1.5.  Replace "," with "." for feature 'add_ons'
1.6.  Convert all currency values to euros for:
	  'monthly_spend_estimated','offer_cost_eur','historic_revenue_12m','revenue_next_12m_observed'.
1.7.  Transform the two formats into a single one for the donation_share_charity feature.
1.8.  Remove the '+' from the web_sessions_90d_raw feature; 
	  to be honest, I don't fully understand what the '+' indicates
1.9.  Remove outliers for 'service_contacts_12m'.
1.10.  Map {'true', 'y', '1', 'yes'} to 1, and {'false', 'f', '0', 'no'} to 0 for:
	  'treatment_sent_flag', 'churned'
1.11. Replace "_" with " " for feature 'baseline_churn_risk_band'
1.12. Change str datatype to int for:
	  'participant_age','web_sessions_90d_raw'.
1.13. Further feature engineering	
	    '''Only consider 3 situations: 
        'campaign_cohort'='Control' and 'treatment_sent_flag'=0;
        'campaign_cohort'='Variant_A' and 'treatment_sent_flag'=1;
        'campaign_cohort'='Variant_B' and 'treatment_sent_flag'=1;'''	
		Remove features: 'legacy_system_id','subscription_date','postcode_area','observation_end_date'	  
1.14. Examine the distribution of features with numeric values
1.15. Save the cleaned data to retention_model_data.csv
---	  

## Part B — Predictive Modeling 
(`src/models.py`)
	### 1. Churn Propensity Model
1.1. The models used to predict `churned`:
	 KNN, LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
1.2. The features not condidered:
	 'customer_id','subscription_id','campaign_cohort','treatment_sent_flag','revenue_next_12m_observed'
1.3. Perform encoding on categorical variables
	 Label encodling for ordinal data: 'baseline_churn_risk_band'
	 One hot encoding for nominal data: 'marketing_channel','country_code'
1.4. Handle imbalanced data by oversampling (in the training set)
1.5. Standardize features
1.6. Grid search for hyperparameter tunning, (5 folds cross-validation is used)
1.7. Save the trained models, prediction accuracy (ROC AUC, PR AUC)
1.8. Explore the fearure importance

### 2. Customer Value / LTV Approximation
2.1. Lifetime Value (LTV)=(Average Yearly Revenue per User) × (Gross Margin %) × (Expected Lifetime in Years)
2.2  The feature monthly_spend_estimated is used to calculate the customer’s LTV for the previous 12 months, 
	 while revenue_next_12m_observed provides the LTV for the next 12 months. 
	 Averaging these two measurements offers a more robust estimate of customer value.
---

## Part C — Causal & Uplift Analysis 
(`src/causal.py`)
	### 1. A/B Test Analysis
1.1. Check the number of customers in each group: Variant A, Variant B, and Control.
1.2. Check the distribution of 'churned'.
1.3. A two-proportions z-test will be used. 
	 It's a statistical hypothesis test that determines if there's a significant difference 
	 between the proportions (percentages) of a binary outcome (yes/no, success/fail) in two independent groups.
1.4. Find: 
	For Variant A vs Control, p-value is 0.0. So the difference between Control and Variant A is significant.
	For Variant B vs Control, p-value is 0.893. So the difference between Control and Variant B is not significant.

### 2. Treatment Effect Heterogeneity (Segments)
2.1.  Participant_age and baseline_churn_risk_band are used separately to group customers, 
	 and a two-proportions z-test is then applied within each group.
2.2. Find:
	 Overall, the results show that experiment variant_a is effective in reducing churn. 
	 Customers aged above 50 appear to benefit more from variant_a than younger customers. 
	 However, when the legacy rule-based churn risk (baseline_churn_risk_band) is very high, 
	 variant_a no longer shows a meaningful effect.
	 
### 3. Causal Estimation Uplift Modeling
3.1. Build an two-model approach to estimate individual treatment effects.
	 A/B testing provides general insights, such as whether Variant A is likely to influence customers overall.
	 However, it does not identify which specific customers are most likely to be affected.
	 In contrast, a two-model causal estimation approach can estimate impact at the individual level, 
	 allowing us to determine which customers are most likely to respond. 
	 This information is highly valuable for more targeted and effective decision-making.
---

## Part D (OPTIONAL-BONUS) — Targeting Policy & ROI 
(`notebooks/analysis.ipynb` or `src/policy.py`)
### 1. Define a Targeting Rule
### 2. Budget Constraint
### 3. ROI Calculation
	 My strategy is to target the top 20% of customers (20% of N, where N is the total customer base). 
	 I prioritize using Variant A, since it is both more effective and less costly than Variant B.
	 First, I rank customers by ['ITE_variant_a', 'ITE_variant_b'] in ascending order. 
	 From this ranking, I select the top 20% of customers. 
	 For those selected, if a customer’s ITE_variant_a is below a chosen threshold (e.g., –0.3), 
	 the customer is targeted with Variant A. 
	 If ITE_variant_a is above the threshold, 
	 I then compare ITE_variant_a and ITE_variant_b and choose the treatment with the lower value. 
	 For example, under this rule, if ITE_variant_a > ITE_variant_b, the customer should be targeted with Variant B.

---


## Part E — Engineering & Reproducibility

### 1. Configuration Management (`configs/`)
* Completed
### 2. Project Structure & Entry Points
* Completed
### 3. (OPTIONAL-BONUS)  Unit Tests (`tests/`)
* Write at least **2–3 unit tests** for:
    * Currency parsing and conversion.
    * Date parsing.
    * Parsing of `engagement_history`.
    * and others...
* Tests should run with a single command (e.g., `pytest`).
### 4. Versioning & Environment
* Completed

---

## Part F — Documentation

### 1. Design Document (`design_doc.md`)
* Completed
### What to expect at the review session:
You will go over your solution with our data scientists (2) where you will have the 
chance to explain your design choices and answer questions from the group. 
* Questions about overall architecture and pipeline.
* Clarifying questions on **key technical decisions** (e.g., choice of uplift model, 
  handling dirty data).
* Questions on model and analysis quality, risks and performance.
### Do you need to prepare presentation?:
* Completed