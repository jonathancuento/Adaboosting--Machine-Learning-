import numpy as np
import pandas as pd

dataset = pd.read_csv("C:/Users/Dell/Desktop/dx.csv")
X = dataset.iloc[:,[0,1]].values
y = dataset.iloc[:,1].values


X_test = np.zeros((18,2))
X_train = np.zeros((len(X)-18, 2))
y_test = np.zeros(18)
y_train = np.zeros(len(X)-18)

for i in range(len(X_train)):
    X_train[i][0] = X[i][0]
    X_train[i][1] = X[i][1]
    y_train[i] = y[i]
for i in range(len(X_test)):
    X_test[i][0] = X[i+len(X_train)][0]
    X_test[i][1] = X[i+len(X_train)][1]
    y_test[i] = y[i+len(X_train)]
    

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#HASTA AQUI SE CREAN LOS DATOS (x,y)




#supongo que desde aqui irá el for
from sklearn import tree
classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred1 = classifier.predict(X_test)


X_train1 = np.zeros((len(X_test),2))

for i in range(len(X_test)):
    X_train1[i][1] = X_test[i][1]

w = np.ones(len(X_test))

p = np.zeros(len(w))
LAvg = 0.6
while LAvg >= 0.5:    
    for i in range(len(w)):
        p[i] = (w[i]/np.sum(w))

    classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
    classifier.fit(X_train1, y_pred1, sample_weight=p)
    y_pred = classifier.predict(X_test)
    
    
    error_vec = np.zeros(len(y_test))
    L = np.zeros(len(y_test))
    thrhold = 0.04
    
    for i in range(len(error_vec)):
        error_vec[i] = np.absolute(((y_pred[i] - y_test[i])/y_test[i]))
        if error_vec[i] > thrhold:
            L[i] = 1
        else:
            L[i] = 0
    LAvg = 0
    for i in range(len(L)):
        LAvg = LAvg + (L[i]*p[i])
    
    b = np.log(1/LAvg)
    for i in range(len(w)):
        w[i] = (w[i] * pow(b, 1-L[i]))

print("Y de testeo")
for i in range(len(y_test)):
    print(y_test[i])
print("Y de prediccion")
for i in range(len(y_pred)):
    print(y_pred[i])