#Leonardo Gracida Munoz A01379812
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as sm
import pickle

df = pd.read_csv("mushrooms.csv")

print("Dataset Shape:")
print(df.shape)

print("Datos vacios de la dataset: ")
print(df.isna().sum())

print("Tipo de dato de cada columna: ")
print(df.dtypes)

for i in df:
    print("Columna: ",i,", Numero de clases: ",len(df[i].value_counts()))

le = LabelEncoder()
for i in df:
    df[i]=le.fit_transform(df[i]) 

X_train, X_test, y_train, y_test = train_test_split(df.drop(["class"],axis=1), df["class"], test_size=0.30, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

print("Train shape, X: ", X_train.shape,", Y: ",y_train.shape)
print("Validation shape, X: ", X_val.shape,", Y: ",y_val.shape)
print("Test shape, X: ", X_test.shape,", Y: ",y_test.shape)

rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=20, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

pred_val = rnd_clf.predict(X_val)
pred = rnd_clf.predict(X_test)

print("Train acc: ",rnd_clf.score(X_train,y_train))
print("Val acc: ",sm.accuracy_score(y_val.to_numpy(),pred_val))
print("Test acc: ",sm.accuracy_score(y_test.to_numpy(),pred))
print("Predicciones: ")
for i in range(15):
    real = 0
    predic = 0
    if y_test.to_numpy()[i] == 1.0:
        real = "Venenoso"
    else:
        real = "Comestible"
    if pred[i] == 1.0:
        predic = "Venenoso"
    else:
        predic = "Comestible"
    print("Real: ",real," ,Prediccion: ", predic)
