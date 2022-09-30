#Lasso Regresyonu
#Ridge resgresyonla aynı amaçtır.
#Ridge ve Lasso arasındaki en büyük formülleri
#Ridgede formülde alphadan sonraki terim kare iken
#Lassoda bu terim mutlak değerdir.
#Ridge de gereksiz bile olsa tüm değişkenler bulunurdu ancak
#Lasso regrsyonda lambda ifadesi yeteri kadar büyük olursa 
#bu değişkenlerin katsasyılarını sıfır yapar.
#Lasso regresyonunda çıkış noktası burasıdır.Gereksiz değişkenleri
#ortadan kaldırmak.
#Ridge ya da lasso birbirinden üstün değildir.

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,LassoCV


#veri seti ve test seti oluşturma

df=pd.read_csv("./Hitters.csv")
df=df.dropna()
dms=pd.get_dummies(df[["League","Division","NewLeague"]])
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

df.head()
df.shape


#Lasso Modeli Kurma

lasso_model=Lasso().fit(X_train,y_train)

lasso_model.alpha
lasso_model.intercept_
lasso_model.coef_

#farklı lambda değerlerine karşılık katsayılar

lasso=Lasso()
coefs=[]
alphas=10**np.linspace(10,-2,100)*0.5

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train,y_train)
    coefs.append(lasso.coef_)

ax=plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")

#☺Tahmin

y_pred=lasso_model.predict(X_train)[0:5] #eğitim setindeki bağımsız değişkenleri kullaranak bağımlı değişkenleri tahmin etmeye çalışıyor
y_pred=lasso_model.predict(X_test)[0:5] #test setindeki bağımsız değişkenleri kullaranak bağımlı değişkenleri tahmin etmeye çalışıyor

##Optimize edilmemiş modelin Hataları##
#Test Hatası

y_pred=lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#R2 Skoru
r2_score(y_test,y_pred)


#Model Tuning
#optimum lambda değerini bulmak için lassocv kullanılacak

lasso_cv_model=LassoCV(cv=10,max_iter=100000).fit(X_train,y_train)
lasso_cv_model.alpha_

lasso_tuned=Lasso().set_params(alpha=lasso_cv_model.alpha_).fit(X_train,y_train)
lasso_tuned=Lasso(alpha=lasso_cv_model.alpha_).fit(X_train,y_train)

y_pred=lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

pd.Series(lasso_tuned.coef_,index=X_train.columns)



































































