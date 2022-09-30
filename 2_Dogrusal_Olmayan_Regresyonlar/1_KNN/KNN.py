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
