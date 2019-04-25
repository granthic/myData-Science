from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

url = "./thefile.csv"
#url = "./data.csv"
#names = ['Year','Name', 'Price', 'Mileage', 'Body Type', 'Description']
names = ['Year','Brand', 'Model', 'Mileage','Engine','Color','Type','Description','MOT','Price']

df = pd.read_csv(url, names=names)

df.describe()

#for name in names:
#print ([name]," : ",df[name].unique())

# Convert the color column to one binary column for each color
df_colors = df['Color'].str.get_dummies().add_prefix('Color: ')
# Convert the type column to one binary column for each type
df_type = df['Type'].apply(str).str.get_dummies().add_prefix('Type: ')
df_Engine = df['Engine'].apply(str).str.get_dummies().add_prefix('Engine: ')

df_Brand = df['Brand'].apply(str).str.get_dummies().add_prefix('Brand: ')


# Add all dummy columns
#df = pd.concat([df,df_Brand, df_type], axis=1)
df = pd.concat([df,df_type], axis=1)
# And drop all categorical columns
df = df.drop(['Description','Engine','Type', 'Color','Brand'], axis=1)

df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

df['Price'].fillna(value=df['Price'].mean(), inplace=True)

df['Mileage_Log_Log'] = np.log(np.log(df['Mileage']))
print(df.describe())
#print(df)
df.drop_duplicates(subset=None, keep='first', inplace=False)
#print(df)
#print(df.describe()) # Mean avg and other functions 
#print(df['Mileage','Price','Type:   AWD'])
print(df.head())

#sns.pairplot(df[['Year','Mileage_Log_Log','MOT']], height=2.0)
#plt.show()

#### 


#df["Price"].hist(bins = 50, log = True)

matrix = df.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, vmax=0.7,cmap='coolwarm',annot=True, square=True)
plt.title('Car Price Variables')

plt.show()


#X = df[['Year', 'Brand', 'Mileage']]
#y = df['Price'].values.reshape(-1, 1)

### Step 4: The Ask Price function
#X = df[['Year', 'Mileage_Log_Log']]
X = df[['Year','MOT', 'Mileage_Log_Log']]
y = df['Price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_normalizer = StandardScaler()
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)

#model = MLPRegressor(hidden_layer_sizes=(100, 100), random_state=42)
model = MLPRegressor(hidden_layer_sizes=(100, 100 ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
model.fit(X_train, y_train)


print("##Now we can predict prices:")
##Now we can predict prices:
y_pred = model.predict(X_test)

y_pred_inv = y_normalizer.inverse_transform(y_pred)
y_test_inv = y_normalizer.inverse_transform(y_test)

# Build a plot
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Prediction #')
plt.ylabel('Real value %')


print("## Now add the perfect prediction line")
# Now add the perfect prediction line
diagonal = np.linspace(15000,20000, 10000)
#diagonal = np.linspace(500, 1500, 100)

plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted ask price ($)')
plt.ylabel('Ask price ($)')
plt.show() 


##Distance traveled (odometer) versus ask price
distances = np.linspace(150000, 250000, 10000)

df = pd.DataFrame([
{
    'Year': 2017,
    'Mileage': mileage,
    'MOT': 1
}
for mileage in distances])

df['Mileage_Log_Log'] = np.log(np.log(df['Mileage']))
X_custom = df[['Year', 'Mileage_Log_Log', 'MOT']]
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



##Predicted ask price versus Construction year
construction_years = list(range(2013,2018 + 1))
df = df
df = pd.DataFrame([
{
    'Year': construction_year,
    'Mileage': 35000,
    'MOT': 1
}
for construction_year in construction_years])

df['Mileage_Log_Log'] = np.log(np.log(df['Mileage']))

print(df)
#X_custom = df[['Year', 'Price', 'Mileage_Log_Log']]
X_custom = df[['Year', 'MOT', 'Mileage_Log_Log']]
print(X_custom)
X_custom = X_normalizer.transform(X_custom)
#X_custom = X_normalizer.fit_transform(X_custom)
print(X_custom)
y_pred = model.predict(X_custom)
price_prediction = y_normalizer.inverse_transform(y_pred)

fig, ax = plt.subplots(1, 1)
ax.plot(construction_years, price_prediction)
plt.xticks(construction_years, construction_years)
ax.set_xlabel('Construction year')
ax.set_ylabel('Predicted ask price (€)')
plt.title('Predicted ask price versus Construction year')
plt.show()

## Periodical check-up (MOT) versus ask price
days_until_MOT = np.linspace(-365, 365, 100)

df = pd.DataFrame([
    {
        'Year': 2018,
        'Mileage': 52000,
        'MOT': days
    }
for days in days_until_MOT])

df['Mileage_Log_Log'] = np.log(np.log(df['Mileage']))
X_custom = df[['Year', 'MOT', 'Mileage_Log_Log']]
X_custom = X_normalizer.transform(X_custom)

y_pred = model.predict(X_custom)
price_prediction = y_normalizer.inverse_transform(y_pred)

fig, ax = plt.subplots(1, 1)
ax.plot(days_until_MOT, price_prediction)
ax.set_xlabel('MOT')
ax.set_ylabel('Predicted ask price (€$)')
ax.set_xlim(0, 365)
plt.title('Predicted ask price versus Days until MOT')



## What is the value of my Peugeot?
df = pd.DataFrame([
    {
        'Year': 2018,
        'MOT':1,
        'Mileage': 44000,
    }
])

df['Mileage_Log_Log'] = np.log(np.log(df['Mileage']))
X_custom = df[['Year', 'MOT', 'Mileage_Log_Log']]
X_custom = X_normalizer.transform(X_custom)

y_pred = model.predict(X_custom)
price_prediction = y_normalizer.inverse_transform(y_pred)
print('Predicted ask price: $%.2f' % price_prediction)

