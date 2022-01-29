import numpy as np
import pandas as pd
import time
import datetime
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de acciones')
    user_input = st.text_input('Introducir TICKER de la empresa' , 'TSLA')
    st.title('Model - LONG-SHORT TERM MEMORY')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2019, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 20, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    data = pd.read_csv(query_string)
    
    # Seleccion de datos
    data['highest hight'] = data['High'].rolling(window=10).max()
    data['lowest low'] = data['Low'].rolling(window=10).min()
    data['trigger'] = np.where(data['High']==data['highest hight'],1,np.nan)
    data['trigger'] = np.where(data['Low']==data['lowest low'],0,data['trigger'])
    data['position'] = data['trigger'].ffill()
    data = data.drop(data.index[[0,1,2,3,4,5,6,7,8,9]])

    # Eleccion de datos
    df = data.drop(['Date','Adj Close','Volume','highest hight', 'lowest low', 'trigger'],axis=1)
    df = df.dropna()
    
    # Describiendo los datos
    st.subheader('Datos del 2010 al 2022') 
    st.write(df.describe())

    # Separacion de datos de entrenamiento y prueba
    X = df.drop('position',axis=1)
    y = df['position']
    X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=101)

    
    modelo = Sequential()
    modelo.add(LSTM(units=na, input_shape=dim_entrada)) #se especifica el num de neuronas
    modelo.add(Dense(units=dim_salida))
    modelo.compile(optimizer='rmsprop', loss='mse')
    modelo.fit(X_train,Y_train,epochs=20,batch_size=32)
    st.subheader('Score del modelo') 
    st.success(modelo.score(X_test,y_test))
    
    # continuar con su codigo
