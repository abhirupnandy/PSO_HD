#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:03:16 2019

@author: abhirup
"""

#      Adding libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import AgglomerativeClustering,KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#    Importing  and Transforming Data

colm=['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
          'exang','oldpeak','slope','ca','thal','num']
data = pd.read_csv("processed_data/processed_cleveland_data.csv",names=colm )
data3 = pd.read_csv("processed_data/processed_va_data.csv",names=colm )
data4 = pd.read_csv("processed_data/processed_switzerland_data.csv",names=colm )
data = data.append(data3,ignore_index=True)
data = data.append(data4,ignore_index=True)
data = data.replace("?",np.nan)
data = data.astype("float64")
#X= data.iloc[:,:].values
X = data.iloc[:, :-1].values
y = data.iloc[:, 13].values
y = np.array([1 if i > 0 else 0 for i in y])
colm = colm[:-1]
imputer = SimpleImputer(missing_values = np.nan,strategy="constant",fill_value = 3)
imputer = imputer.fit(X)
X = imputer.transform(X)
X = pd.DataFrame(X,columns =colm)

#   Creating vectors
n = 100
b_vec = np.random.randint(2,size=(n,13))
sum_1=np.zeros(n)
for i in enumerate(b_vec):
    sum_1[i[0]] = np.sum(np.where(i[1] == 1))

coords = (np.random.rand(100, 2)*0.1 )

w, c1, c2,  epoch = 0.5, 0.3, 0.8, 500
r1 = np.random.rand(n,1)*0.001
r2 = np.random.rand(n,1)*0.001
CP = np.zeros(n)

def func(x):
    return (10*(1 - x)**2 + .1*(1 - x**2)**2)

CP = 10*(r1 - 0.5)
V = r2 - 0.5
CF = func(CP)

LBP , LBF = CP , CF
GBF = np.max(LBF)
GBP = LBP[np.where(LBF == GBF)]
for i in range(epoch):
    V = w*V  + c1*r1*(LBP - CP) + c2*r2*(GBP - CP)
    CP = CP + V
    CF = func(CP)
    LBF1 = np.maximum(CF,LBF)
    GBF = np.max(LBF1)
    GBP = LBP[np.where(LBF1 == GBF)]
    for i in range(n):
        if LBF1[i] in CF[i]:
            LBP[i]=CP[i]
        else:
            LBP[i]=LBF[i]

s_list= np.zeros((n,2))
for i in enumerate(sum_1*.1):
    s_list[i[0]] = [i[1], GBP]
   
par_list = np.flip(cosine_similarity(s_list,coords).diagonal().argsort()[-25:-1])

"""print("Agglomerative")
sum = 0
a_list = np.zeros((len(par_list),2))
for i in enumerate(par_list):
    att = np.where(b_vec[i[1]] == 1)
    att=att[0]
    col = []
    for j in att:
        col.append(colm[j-1])
    X_new = X.drop(col,axis=1)
    clustering = AgglomerativeClustering().fit(X_new)
    y_pred = clustering.labels_
    a_list[i[0]][0] = accuracy_score(y_pred,y)
    a_list[i[0]][1] = X_new.shape[1]
    sum +=accuracy_score(y_pred,y)
a = a_list[:,0].argmax()
print(a_list[a])
print(sum/len(par_list))

print(classification_report(y_pred,y))
sum = 0
a_list = np.zeros((len(par_list),2))

for i in enumerate(par_list):
    clustering = AgglomerativeClustering().fit(X)
    y_pred = clustering.labels_
    sum +=accuracy_score(y_pred,y)
print(sum/len(par_list))
print(classification_report(y_pred,y))"""

"""print("KMeans")
sum = 0
a_list = np.zeros((len(par_list),2))
for times in range(iter):
    for i in enumerate(par_list):
        att = np.where(b_vec[i[1]] == 1)
        att=att[0]
        col = []
        for j in att:
            col.append(colm[j-1])
        X_new = X.drop(col,axis=1)
        clustering = KMeans().fit(X_new)
        y_pred = clustering.labels_
        a_list[i[0]][0] = accuracy_score(y_pred,y)
        a_list[i[0]][1] = X_new.shape[1]
        sum +=accuracy_score(y_pred,y)

a=a_list[:,0].argmax()
print(a_list[a])
print(sum/len(par_list)/iter)
sum = 0
a_list = np.zeros((len(par_list),2))
for i in enumerate(par_list):
    clustering = KMeans().fit(X)
    y_pred = clustering.labels_
    sum +=accuracy_score(y_pred,y)
print(sum/len(par_list))"""

"""print("GAUSSIAN NB")
sum = 0
a_list = np.zeros((len(par_list),2))
for times in range(iter):
    for i in enumerate(par_list):
        att = np.where(b_vec[i[1]] == 1)
        att=att[0]
        col = []
        for j in att:
            col.append(colm[j-1])
        X_new = X.drop(col,axis=1)
        clustering = GaussianNB(  ).fit(X_new,y)
        y_pred = clustering.predict(X_new)
        a_list[i[0]][0] = accuracy_score(y_pred,y)
        a_list[i[0]][1] = X_new.shape[1]
        sum +=accuracy_score(y_pred,y)

a = a_list[:,0].argmax()
#print(a_list[a])
print(sum/len(par_list)/iter)

print(classification_report(y_pred,y))
sum = 0
a_list = np.zeros((len(par_list),2))
for times in range(iter):
    for i in enumerate(par_list):
        clustering = GaussianNB().fit(X,y)
        y_pred = clustering.predict(X)
        a_list[i[0]][0] = accuracy_score(y_pred,y)
        a_list[i[0]][1] = X_new.shape[1]
        sum +=accuracy_score(y_pred,y)

a = a_list[:,0].argmax()
#print(a_list[a])
print(sum/len(par_list)/iter)

print(classification_report(y_pred,y))"""

"""print("MLP Classifier")
sum = 0
a_list = np.zeros((len(par_list),2))
for i in enumerate(par_list):
    att = np.where(b_vec[i[1]] == 1)
    att=att[0]
    col = []
    for j in att:
        col.append(colm[j-1])
    X_new = X.drop(col,axis=1)
    X_new_train,X_new_test,y_train,y_test = train_test_split(X_new,y,test_size =0.33)
    clf = MLPClassifier(hidden_layer_sizes = (14,28,56,28,14),
                        activation = 'tanh',
                        max_iter = 1000,
                        learning_rate_init = 0.01).fit(X_new_train,y_train)
    y_pred = clf.predict(X_new_test)
    a_list[i[0]][0] = accuracy_score(y_pred,y_test)
    a_list[i[0]][1] = X_new.shape[1]
    sum +=accuracy_score(y_pred,y_test)
a = a_list[:,0].argmax()
print(a_list[a])
print(sum/len(par_list))

print(classification_report(y_pred,y_test))
sum = 0
a_list = np.zeros((len(par_list),2))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.33)
clf = MLPClassifier(hidden_layer_sizes = (14,28,56,28,14),
                        activation = 'tanh',
                        max_iter = 1000,
                        learning_rate_init = 0.01).fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))"""

print("Random Forest")
sum = 0
a_list = np.zeros((len(par_list),2))
for i in enumerate(par_list):
    att = np.where(b_vec[i[1]] == 1)
    att=att[0]
    col = []
    for j in att:
        col.append(colm[j-1])
    X_new = X.drop(col,axis=1)
    X_new_train,X_new_test,y_train,y_test = train_test_split(X_new,y,test_size =0.33)
    clf = RandomForestClassifier(n_estimators = 50).fit(X_new_train,y_train)
    y_pred = clf.predict(X_new_test)
    a_list[i[0]][0] = accuracy_score(y_pred,y_test)
    a_list[i[0]][1] = X_new.shape[1]
    sum +=accuracy_score(y_pred,y_test)
a = a_list[:,0].argmax()
print(a_list[a])
print(sum/len(par_list))

print(classification_report(y_pred,y_test))
sum = 0
a_list = np.zeros((len(par_list),2))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.33)
clf = RandomForestClassifier(n_estimators = 50).fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))