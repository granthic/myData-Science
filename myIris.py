from sklearn.datasets import load_iris


iris=load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.target[0])
print(iris.data[0])