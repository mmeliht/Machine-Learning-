#Amaç hata kareler ortalaması gibi değerleri en minimum
#seviyede tutmaya çalışmaktır.
#Bunu da katsayılara ceza uygulayarak yapmaya çalışır.



import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

#veri seti bir beyzbol takımının oyuncularına ait
#veriler kullanılarak maaş tahmin etme


#verileri okuma ve düzenleme

df=pd.read_csv("./Hitters.csv")
df=df.dropna()
dms=pd.get_dummies(df[["League","Division","NewLeague"]])

y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=42)

df.head()
df.shape


#Ridge model oluşturma
ridge_model=Ridge(alpha=5).fit(X_train,y_train)

ridge_model.coef_
ridge_model.intercept_

#alpha değerleri için rastgele sayılar dizisi oluşturuyoruz.
lambdalar=10**np.linspace(10,-2,100)*0.5

katsayilar=[]

for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train,y_train)
    katsayilar.append(ridge_model.coef_)

#lambdalara karşılık gelen katsilar grafiği
ax=plt.gca()
ax.plot(lambdalar,katsayilar)
ax.set_xscale("log")    


#Tahmin yapma 

ridge_model=Ridge().fit(X_train,y_train)
y_pred=ridge_model.predict(X_train)

y_pred[0:10]
y_train[0:10]

#hata kareler ortalmasının kare kökü train setine göre
RMSE=np.sqrt(mean_squared_error(y_train, y_pred))
RMSE

#cross validation hesabı
np.sqrt(np.mean(-cross_val_score(ridge_model, X_train,y_train, cv=10, scoring="neg_mean_squared_error")))


#test hatası bulma
y_pred=ridge_model.predict(X_test)
RMSE_test=np.sqrt(mean_squared_error(y_test, y_pred))
RMSE_test

#cross validation hesabı
np.sqrt(np.mean(-cross_val_score(ridge_model, X_test,y_test, cv=10, scoring="neg_mean_squared_error")))


#Model Tuning

ridge_model=Ridge(1).fit(X_train,y_train)
y_pred=ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

lambdalar1=np.random.randint(0,1000,100)
lambdalar2=10**np.linspace(10,-2,100)*0.5

#lambdalar2 için
ridgecv=RidgeCV(alphas=lambdalar2, scoring="neg_mean_squared_error", cv=10, normalize=True)
ridgecv.fit(X_train,y_train)

ridgecv.alpha_

#final modeli

ridge_tuned=Ridge(alpha=ridgecv.alpha_).fit(X_train, y_train)

#final test hatası
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


#lambdalar1 için

ridgecv=RidgeCV(alphas=lambdalar1, scoring="neg_mean_squared_error", cv=10, normalize=True)
ridgecv.fit(X_train,y_train)

ridgecv.alpha_
#final modeli

ridge_tuned=Ridge(alpha=ridgecv.alpha_).fit(X_train, y_train)

#final test hatası
y_pred=ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

















































