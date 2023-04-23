# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
# STEP 1:
Read the given Data

# STEP 2:
Clean the Data Set using Data Cleaning Process
 
# STEP 3:
Apply Feature Transformation techniques to all the features of the data set

# STEP 4:
Print the transformed features

# PROGRAM:
```
# NAME: R.K PRAGALYAA SHREE
# REG NO: 212221040125

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
## OUTPUT:
![ds 1](https://user-images.githubusercontent.com/128135934/233828538-f213d833-4a9a-4e0c-8e45-38ca807224de.png)

![ds -2](https://user-images.githubusercontent.com/128135934/233828559-13514222-d8c4-401f-b0bd-8a13e924a92f.png)

![ds3](https://user-images.githubusercontent.com/128135934/233828575-115ebdb8-9305-4def-b9c1-3198fc5c0b4f.png)

![ds 4](https://user-images.githubusercontent.com/128135934/233828642-34a679cc-46c9-4fac-b10b-4d18e415f647.png)

![ds 6](https://user-images.githubusercontent.com/128135934/233828667-c009b46b-55e1-4a75-a765-be8769ce41c4.png)

![ds 7](https://user-images.githubusercontent.com/128135934/233828695-574b41ed-de59-46a9-af04-7afe430b5410.png)

![ds 8](https://user-images.githubusercontent.com/128135934/233828709-fe4cbbd3-fdd6-4e73-8f15-3baf4aad4ba1.png)

![ds 9](https://user-images.githubusercontent.com/128135934/233828727-bfb93e68-28e6-420a-a0e1-88b27440c6e6.png)

![ds 10](https://user-images.githubusercontent.com/128135934/233828729-714334ef-5aec-41ce-8e39-8babb19aa55e.png)

# RESULT:
Thus feature transformation is done for the given dataset.






