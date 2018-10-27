from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
# print(iris.target_names)
# print(type(iris.data))
# print(iris.feature_names)
# print(iris.data[0:5])
# print(iris.target)
data = pd.DataFrame({'sepal length':iris.data[:,0],'sepal width':iris.data[:,1],'petal length':iris.data[:,2],'petal width':iris.data[:,3],'species':iris.target})

# print(data.head())
X=data[['sepal length','sepal width','petal length','petal width']]
Y=data['species']
# print(X.head())
# print(Y.shape)
X_train ,X_test,Y_train ,Y_test = train_test_split(X,Y,test_size=0.3)

new = RandomForestClassifier(n_estimators=100)



new.fit(X_train,Y_train)

features = pd.Series(new.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
print(features)
Y_pred = new.predict(X_test)
print("Accuracy",metrics.accuracy_score(Y_test,Y_pred))