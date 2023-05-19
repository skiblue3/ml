import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('url.csv')

X = data.drop(columns=['Domain','Label'],axis = 1)
Y = data['Label']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
GradBoost = GradientBoostingClassifier(init=None, learning_rate=0.1, n_estimators=200)
GradBoost.fit(X_train, Y_train)
Y_pred_boost = GradBoost.predict(X_test)

accuracy_boost = metrics.accuracy_score(Y_test, Y_pred_boost)
print("Accuracy of Gradient Boosted model: ", accuracy_boost*100)


