#Leonardo Gracida Munoz A01379812
#Importamos todas las librerias necesarias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import sklearn.metrics as sm

#Leemos la dataset de los hongos
df = pd.read_csv("mushrooms.csv")

#Mostramos las caracteristicas de la dataset
print("Dataset Shape:")
print(df.shape)

print("Datos vacios de la dataset: ")
print(df.isna().sum())

print("Tipo de dato de cada columna: ")
print(df.dtypes)

#Como son puros datos categoricos mostramos cuantas clases tiene cada uno
for i in df:
    print("Columna: ",i,", Numero de clases: ",len(df[i].value_counts()))

"""Como son muchas clases categoricas por columnas usamos label encoder,
ya que note que los hongos se pueden clasficiar rapido o no son muy diferentes entre si
los hongos venenosos entre si, como tampoco lo son los comestibles entre si"""

#Paasamos las columnas de calores categoricos a valores numericos con el label encoder
le = LabelEncoder()
for i in df:
    df[i]=le.fit_transform(df[i]) 

#Dividimos la dataset en train, test y validation.
X_train, X_test, y_train, y_test = train_test_split(df.drop(["class"],axis=1), df["class"], test_size=0.30, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

#Mostramos la forma de cada una de las separacaiones del dataset
print("Train shape, X: ", X_train.shape,", Y: ",y_train.shape)
print("Validation shape, X: ", X_val.shape,", Y: ",y_val.shape)
print("Test shape, X: ", X_test.shape,", Y: ",y_test.shape)

#Creamos el modelo de RandomForest
rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=20, n_jobs=-1, random_state=42)
#Entrenamos el modelo
rnd_clf.fit(X_train, y_train)

#Hacemos las predicciones
pred_val = rnd_clf.predict(X_val)
pred = rnd_clf.predict(X_test)

#Mostramos la presicion de cada una de las particiones del dataset
print("Train acc: ",rnd_clf.score(X_train,y_train))
print("Val acc: ",sm.accuracy_score(y_val.to_numpy(),pred_val))
print("Test acc: ",sm.accuracy_score(y_test.to_numpy(),pred))

#Mostramos algunas predicciones
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
