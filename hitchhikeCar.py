# The imports
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


sns.set()

#url = "./thefile.csv"
url = "./data.csv"

df = pd.read_csv(url)

# Convert the color column to one binary column for each color
df_colors = df['Color'].str.get_dummies().add_prefix('Color: ')
# Convert the type column to one binary column for each type
df_type = df['Type'].apply(str).str.get_dummies().add_prefix('Type: ')

# Add all dummy columns
df = pd.concat([df, df_colors, df_type], axis=1)
# And drop all categorical columns
df = df.drop(['Brand', 'Type', 'Color'], axis=1)

#print(df)
df['Odometer Log Log'] = np.log(np.log(df['Odometer']))

matrix = df.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, vmax=0.7, square=True,annot=True,cmap='coolwarm')
plt.title('Car Price Variables')
plt.show()
sns.pairplot(df[['Construction Year', 'Days Until MOT', 'Odometer Log Log', 'Ask Price']], size=1.5)
plt.show()

#ask Price
X = df[['Construction Year', 'Days Until MOT', 'Odometer Log Log']]
y = df['Ask Price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

X_normalizer = StandardScaler()
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)

model = MLPRegressor(hidden_layer_sizes=(100, 100), random_state=42)
model.fit(X_train, y_train)

# Now we can predict prices:
y_pred = model.predict(X_test)

y_pred_inv = y_normalizer.inverse_transform(y_pred)
y_test_inv = y_normalizer.inverse_transform(y_test)

# Build a plot
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(500, 1500, 100)
plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted ask price (€)')
plt.ylabel('Ask price (€)')
plt.show()




##Distance traveled (odometer) versus ask price
distances = np.linspace(150000, 250000, 10000)

my_car = pd.DataFrame([
    {
        'Construction Year': 1998,
        'Odometer': odometer,
        'Days Until MOT': 150
    }
for odometer in distances])

my_car['Odometer Log Log'] = np.log(np.log(my_car['Odometer']))
X_custom = my_car[['Construction Year', 'Days Until MOT', 'Odometer Log Log']]
X_custom = X_normalizer.transform(X_custom)

y_pred = model.predict(X_custom)
price_prediction = y_normalizer.inverse_transform(y_pred)

fig, ax = plt.subplots(1, 1)
ax.plot(distances, price_prediction)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_xlabel('Distance travelled (km)')
ax.set_ylabel('Predicted ask price (€)')
plt.title('Predicted ask price versus Distance travelled')
plt.show()

my_car = pd.DataFrame([
    {
        'Construction Year': 1998,
        'Odometer': 220000,
        'Days Until MOT': 150
    }
])

my_car['Odometer Log Log'] = np.log(np.log(my_car['Odometer']))
X_custom = my_car[['Construction Year', 'Days Until MOT', 'Odometer Log Log']]
X_custom = X_normalizer.transform(X_custom)

y_pred = model.predict(X_custom)
price_prediction = y_normalizer.inverse_transform(y_pred)
print('Predicted ask price: €%.2f' % price_prediction)