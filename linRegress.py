import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

url = "./thefile.csv"
#url = "./data.csv"
#names = ['Year','Name', 'Price', 'Mileage', 'Body Type', 'Description']
names = ['Year','Brand', 'Model', 'Mileage','Engine','Color','Type','Description','MOT','Price']

df = pd.read_csv(url, names=names)
df = df.dropna()

df = df.drop(['Brand','Model','Engine','Color','Description'], axis=1)
X = df.drop('Price', axis=1)
y = df[['Price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=1)

reg = LinearRegression()
reg.fit(X_train[['Mileage']], y_train)

y_predicted = reg.predict(X_test[['Mileage']])

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
#plt.show()


diagonal = np.linspace(15000, 20000, 10000)

plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted ask price ($)')
plt.ylabel('Ask price ($)')
plt.show() 
