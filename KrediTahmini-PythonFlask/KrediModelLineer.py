import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np



def get_model():
    data = pd.read_csv('C:/Users/kulas/Desktop/KrediTahmini-PythonFlask/kredi.csv', sep=";")
   
    data.KrediDurumu[data.KrediDurumu == 'krediver'] = 1
    data.KrediDurumu[data.KrediDurumu == 'verme'] = 0
    data.telefonDurumu[data.telefonDurumu == 'var'] = 1
    data.telefonDurumu[data.telefonDurumu == 'yok'] = 0
    data.evDurumu[data.evDurumu == 'evsahibi'] = 1
    data.evDurumu[data.evDurumu == 'kiraci'] = 0
   
    data = data.astype(float)
    data.head()

    X=data.drop('KrediDurumu',axis=1)
    Y=data['KrediDurumu']
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
    
    log=LogisticRegression()
    log.fit(x_train,y_train)
    print(log.score(x_train,y_train))






if __name__ == "__main__":
    get_model()
   