# Survival Analysis

## Kaplan-Meier estimator

- a non-parametric technique of estimating and plotting the survival probability as a function of time.

### Assumptions:
- The event of interest is unambiguous and happens at a clearly specified time (disease).
- The survival probability of all observations is the same, it does not matter exactly when they have entered the study (randomness).
- Censored observations have the same survival prospects as observations that continue to be followed.

### Goal:
approximate true survival function from the collected data.

### Formula:
The survival probability at time t is equal to the product of the percentage chance of surviving at time t and each prior time.

### Approach:
KM curves: a plot of the KM estimator over time. (Can add confidence interval using Greenwood method).

### Explanation of the curve:
The y-axis represents the ```probability``` that the subject still has not experienced the event of interest after surviving up to ```time t```, represented on the x-axis. Each drop in the survival function (approximated by the Kaplan-Meier estimator) is caused by the event of interest happening for at least one observation.

The y-axis drop in magnitute indicates how many observations happened at that time. Esitmate the risk factor at that time.

### Source:
1. https://towardsdatascience.com/introduction-to-survival-analysis-the-kaplan-meier-estimator-94ec5812a97a

2. https://github.com/erykml/medium_articles/blob/master/Statistics/kaplan_meier.ipynb

## Log-rank Test

It is a statistical test that compares the survival probabilities between two groups (or more). The null hypothesis of the test states that there is no difference between the survival functions of the considered groups.

### Goal:
compare the KM curves

#### Additional assumptions:
proportional hazards assumption — the hazard ratio should be constant throughout the study period. (when the plot of two cases cross, the assumption violated)

## Common mistakes:

- Interpreting the ends of the curves

Pay special attention when interpreting the end of the survival curves, as any big drops close to the end of the study can be explained by only a few observations reaching this point of time (this should also be indicated by wider confidence intervals).

- omitted-variable bias

The Kaplan-Meier estimator is a univariable method, as it approximates the survival function using at most one variable/predictor.

## Python examples:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter 
from lifelines.statistics import (logrank_test, 
                                  pairwise_logrank_test, 
                                  multivariate_logrank_test, 
                                  survival_difference_at_fixed_point_in_time_test)

plt.style.use('seaborn')

T = df['tenure'] # time between get diagnoses
E = df['churn'] # 0 or 1 binary represents end with disease

kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)

kmf.plot(at_risk_counts=True) # seaborn interactive tool
plt.title('Kaplan-Meier Curve')

print(kmf.median_survival_time_) #the point in time in which on average 50% of the population has already diagnosed. if returns inf, means we do not observe that point in the data.

# compare with the multiple classes

kmf = KaplanMeierFitter()

for payment_method in df['PaymentMethod'].unique(): #class of paymentmethod
    
    flag = df['PaymentMethod'] == payment_method
    
    kmf.fit(T[flag], event_observed=E[flag], label=payment_method)
    kmf.plot(ax=ax)

plt.title("Survival curves by payment methods");

#Apply log-rank test for two self-picked classes

credit_card_flag = df['PaymentMethod'] == 'Credit card (automatic)'
bank_transfer_flag = df['PaymentMethod'] == 'Bank transfer (automatic)'

results = logrank_test(T[credit_card_flag], 
                       T[bank_transfer_flag], 
                       E[credit_card_flag], 
                       E[bank_transfer_flag])
results.print_summary()

#or all classes one by one

results = pairwise_logrank_test(df['tenure'], df['PaymentMethod'], df['churn'])
results.print_summary()

#or compare between all classes assume all survival curves are identical. Can observed from plot.

results = pairwise_logrank_test(df['tenure'], df['PaymentMethod'], df['churn'])
results.print_summary() 

#see if at a time = 60, the results is significant between groups
results = survival_difference_at_fixed_point_in_time_test(60, 
                                                          T[credit_card_flag], 
                                                          T[bank_transfer_flag], 
                                                          E[credit_card_flag], 
                                                          E[bank_transfer_flag])
results.print_summary()
```

## Limitation:
- We cannot evaluate the magnitude of the predictor’s impact on survival probability.
- We cannot simultaneously account for multiple factors for observations, for example, the country of origin and the phone’s operating system.
- The assumption of independence between censoring and survival (at time t, censored observations should have the same prognosis as the ones without censoring) can be inapplicable/unrealistic.
- When the underlying data distribution is (to some extent) known, the approach is not as accurate as some competing techniques.

