import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
df=load_boston()
dataset=pd.DataFrame(df.data)
dataset.head()
dataset.columns=df.feature_names
dataset["Price"]=df.target
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,45,50,60,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

prediction_ridge=ridge_regressor.predict(x_test)

import seaborn as sns
sns.distplot(y_test-prediction_ridge)