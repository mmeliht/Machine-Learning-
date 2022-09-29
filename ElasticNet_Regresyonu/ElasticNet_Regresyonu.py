#ElastikNet Regresyonu
#ElastikNet L1 ve L2 yaklaşımlarını birlşetirir
#Ridge ve Lassonun birleşimidir.
#Daha etkin bir düzgünleştirme işlemi yapar.
#Ridge tarzı cezalandırma Lasso tarzı değişken seçimi yapar.
#İkisine göre daha dirençlidir.

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV ,ElasticNetCV

#veri seti ve test seti oluşturma

df=pd.read_csv("./Hitters.csv")
df=df.dropna()
dms=pd.get_dummies(df[["League","Division","NewLeague"]])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#Model Kurma

enet_model= ElasticNet().fit(X_train,y_train)

enet_model.coef_ #katsayılara erişme
enet_model.intercept_ #sabit terime erişme

#Tahmin

enet_model.predict(X_train)[0:10] #eğitim setine göre tahmin

enet_model.predict(X_test)[0:10] #test setine göre tahmin

y_pred=enet_model.predict(X_test)

#Test hatası

np.sqrt(mean_squared_error(y_test, y_pred))

#R2 skoru
r2_score(y_test,y_pred)

#Model Tuning

enet_cv_model=ElasticNetCV(cv=10).fit(X_train,y_train)

enet_cv_model.alpha_
enet_cv_model.intercept_
enet_cv_model.coef_

#Final Modeli

enet_tuned=ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train,y_train)

#Test hatası
y_pred=enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))




















































































