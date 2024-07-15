#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score

data = pd.read_csv("Forest_fire.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
log_reg = LogisticRegression()
lin_reg = LinearRegression()


log_reg.fit(X_train, y_train)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print("Accuracy of")
print('Linear Regression:',lin_reg.score(X_test,y_test)*100)
print('Logistic Regression:',log_reg.score(X_test,y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNN algorithm is:",accuracy*100)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest algorithm is:",accuracy*100)


inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


