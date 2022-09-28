#Çoklu Doğrusal Regresyon 

import pandas as pd


df=pd.read_csv("Advertising.csv")
df=df.iloc[:,1:len(df)]
df.head()

X=df.drop("sales",axis=1)
y=df[["sales"]]

X.head()
y.head()

#stat ile model kurma
##stat ile modelde çok daha detaylı verilere ulaşabiliriz.
###Ancak genelde scikit learn kullanırız.
import statsmodels.api as st

lineerModel=st.OLS(y,X)
model=lineerModel.fit()
model.summary()

#scikit learn ile model kurma

from sklearn.linear_model import LinearRegression

lineerModel=LinearRegression()
model=lineerModel.fit(X,y)
model.intercept_  #denklemdeki sabit değişken
model.coef_       #denklemdeki deşkenlerin kat sayıları

#yani denklem Sales=2.94+TV*0.04+radio*0.19-newspaper*0.001
#30 TV 10 radio 40 gazete için
#2.94+30*0.04+10*0.19-40*0.001 =5.9devirli

#modelle sonucu tahmin etme
yeni_veri=[[30],[10],[40]]

yeni_veri=pd.DataFrame(yeni_veri).T
model.predict(yeni_veri)

#hata karaler ortalamasını kodla bulma
from sklearn.metrics import mean_squared_error
y.head() #gerçek değerler

model.predict(X) #modelin tahmin ettiği değerler

MSE=mean_squared_error(y,model.predict(X) ) #hata kareler ortalaması

import numpy as np 
RMSE=np.sqrt(MSE) # hata kareler ortalmasının karekökü



#Model Tuning (Model doğrulama)
# Doğrusal regresyonda hiperparametre olmadığından
#dolayı model doğrulama işlemi yapılacak

X.head() #bağımsız değişken
y.head() #bağımlı değişken

#sınama seti yaklaşımıyla hata hesaplama
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)
X_train.head()

lm=LinearRegression()
model=lm.fit(X_train,y_train)

#eğitim hatası
np.sqrt(mean_squared_error(y_train, model.predict(X_train)))

#test hatası
np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

#☺her seferinde  hangi 20ye 80 lik alanı kullancağı belli olmadığı
#için bu yöntemle hata hesaplamak yanıltıcı olabilir.


#k-katlı cross validation 
from sklearn.model_selection import cross_val_score

#10 parçaya bölerek her defasında 9uyla eğitim yapıp 
#biriyle test etti ve çıkan hata skorları
cross_val_score(model, X_train,y_train, cv=10, scoring="neg_mean_squared_error")


#çıkan hataların ortalaması alınarak ortalama hata hesaplanır (hata kareler üzerinden) mse
np.mean(-cross_val_score(model, X_train,y_train, cv=10, scoring="neg_mean_squared_error"))

#rmse hata kareler ortalamasının kare kökü
np.sqrt(np.mean(-cross_val_score(model, X_train,y_train, cv=10, scoring="neg_mean_squared_error")))









 



















