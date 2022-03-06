import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
warnings.filterwarnings('ignore')

house = pd.read_csv(r'C:\Users\Harish Sundaralingam\Desktop\Stats and ML\Machine Learning\Maison.csv')
print(house.head())

house = house.rename(columns = {'PRIX':'price','SUPERFICIE':'area','CHAMBRES':'rooms',
                               'SDB':'bathroom','ETAGES':'floors','ALLEE':'driveway',
                               'SALLEJEU':'game_room','CAVE':'cellar','GAZ':'gas','AIR':'air',
                               'GARAGES':'garage','SITUATION':'situation'})


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

lm = LinearRegression()

X = house[['area','rooms','bathroom','floors','driveway','game_room',
          'cellar','gas', 'air','garage','situation']]
Y = house[['price']]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

lm.fit(X_train,y_train)

print(lm.coef_)

coef = pd.DataFrame(lm.coef_.T, X.columns, columns = ['Coefficients'])
print(coef)

print('Training Score',lm.score(X_train,y_train))

print('Testing Score',lm.score(X_test, y_test))

predictions = lm.predict(X_test)
print(predictions)

plt.subplots(figsize=(12,12))
sns.scatterplot(y_test['price'],predictions.flatten())
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

print('Mean Asolute Error:', metrics.mean_absolute_error(y_test,predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,predictions))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

import statsmodels.api as sm 

# Unlike sklearn that adds an intercept to our data for the best fit, statsmodel doesn't. We need to add it ourselves
# Remember, we want to predict the price based off our features.
# X represents our predictor variables, and y our predicted variable.
# We need now to add manually the intercepts

X_endog = sm.add_constant(X_test)

res = sm.OLS(y_test,X_endog)
res.fit()

print(res.fit().summary())

X_endog_test = sm.add_constant(X_train)
model = res.fit()
predictions = model.predict(X_endog_test)

plt.figure(figsize=(10,10))
sns.scatterplot(y_train['price'], predictions)
plt.show()