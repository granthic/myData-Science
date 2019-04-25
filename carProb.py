# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns

#sns.set()

# Load dataset
#url = "./iris.csv"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = pandas.read_csv(url, names=names)

#url = "./thefile - Copy (2).csv"
url = "./thefile.csv"
#names = ['Year','Name', 'Price', 'Mileage', 'Body Type', 'Description']
names = ['Year','Maker', 'Model','Price', 'Mileage','Engine','Description']

dataset = pandas.read_csv(url, names=names)

#print(dataset)
print(dataset.head())
#dataset.plot()
#plt.show()
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

sns.pairplot(dataset[['Year', 'Maker', 'Model','Mileage', 'Price']], size=1.5)
plt.show()

#matrix = dataset.corr()
#f, ax = plt.subplots(figsize=(4, 6))
#sns.heatmap(matrix, vmax=0.69, square=True)
#sns.lmplot(x='Mileage',y='Price',data=dataset)
matrix = dataset.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, vmax=0.7, square=True)
plt.title('Car Price Variables')

#pair Plot 
#sns.pairplot(dataset)

plt.title('Car Price Variables')
plt.show()

#print(dataset[['year','Price','Mileage']])
#filteredDataset = dataset['Mileage'] <15000
#print(filteredDataset)

#print(dataset[filteredDataset])
#dataset[filteredDataset].plot(kind='box', subplots=True, layout=(2,2), sharex=True, sharey=False)
#plt.show()
#filteredDataset.plot()
#plt.show()
## histograms
#dataset.hist()
#plt.show()

## scatter plot matrix
#scatter_matrix(dataset)
#plt.show()









print("--SOLVED CARPROB--")