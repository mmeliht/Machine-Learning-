#K en yakın komşu algoritması


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR


from warnings import filterwarnings
filterwarnings("ignore") #uyarı mesajlarını engeller

#veri seti ve test seti oluşturma 
df=pd.read_csv("Hitters.csv")
df=df.dropna() #veri setindeki eksik gözlemleri siler
dms=pd.get_dummies(df[["League","Division","NewLeague"]])#veri setinde bulunan kategorik değişkenleri dami değişkkenine çevirmemiz gerekiyor.
y=df["Salary"]
X_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
X=pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#Model oluşturma
knn_model=KNeighborsRegressor().fit(X_train,y_train)


#Tahmin
y_pred=knn_model.predict(X_test,)
np.sqrt(mean_squared_error(y_test, y_pred))#hata kareler ortalmasının karekökü
#ilkel test hatası hesaplandı

#Model Tuning

RMSE=[]

for k in range(10):
    k+=1
    knn_model=KNeighborsRegressor(n_neighbors=k).fit(X_train,y_train)
    p_pred=knn_model.predict(X_test)
    rmse=np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
   # print("k: ",k,"için RMSE degeri: ",rmse)

#GridSearchCV
#Belirlemeye çalıştığımız hiper parametrelerin değerlerini belirlemeke için kullanılan fonksiyon
#Birbirinden farklı ve fazla parametre olduğunda kullanımı avantajlıdır.
knn_params={"n_neighbors":np.arange(1,30,1)}
knn=KNeighborsRegressor()
knn_cv_model=GridSearchCV(knn, knn_params, cv=10).fit(X_train, y_train)
knn_cv_model.best_params_

#Final Model
knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"]).fit(X_train,y_train)

#Final Model Test Hatası
y_pred=knn_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
































































