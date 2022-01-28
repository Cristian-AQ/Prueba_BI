from doctest import DocFileSuite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
# from pandas_datareader import data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from sklearn import metrics
import streamlit as st


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de tendencia de acciones')
    user_input = st.text_input('Introducir cotización bursátil' , 'MSFT')
    st.title('Model - Support Vector Regresion')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 10, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)

    # Seleccion de datos
    # Obtenemos la data menos la ultima fila
    data = df.head(len(df)-1)

    days = list()
    adj_close_prices = list()
    # Obtenemos la fecha y precios de cierre ajustados
    df_days = data.loc[:, 'Date']
    df_adj_close = data.loc[:, 'Adj Close']

    # Describiendo los datos
    st.subheader('Datos del 2021 de Diciembre') 
    st.write(data.head(10))
    
    
    for day in df_days:
        days.append([int(day.split('-')[2])])

    for adj_close_price in df_adj_close:
        adj_close_prices.append( float(adj_close_price) )



    # Creamos 3 modelos SVR
    # Creamos y entrenamos un modelo SVR usando un kernel lineal
    lin_svr = SVR(kernel = 'linear', C=1000)
    lin_svr.fit(days, adj_close_prices)
    # Creamos y entrenamos un modelo SVR usando un kernel polinomial
    pol_svr = SVR(kernel = 'poly', degree = 2)
    pol_svr.fit(days, adj_close_prices)
    # Creamos y entrenamos un modelo SVR usando un kernel rbf
    rbf_svr = SVR(kernel = 'rbf', C=1000, gamma = 0.15)
    rbf_svr.fit(days, adj_close_prices)

    # Graficamos los modelos cual fue el mejor modelo
    st.figure(figsize=(16,8))
    st.scatter(days, adj_close_prices, color='red', label='Data')
    st.plot(days, rbf_svr.predict(days), color='green', label='Modelo RBF')
    st.plot(days, pol_svr.predict(days), color='orange', label='Modelo Polinomial')
    st.plot(days, lin_svr.predict(days), color='blue', label='Modelo Lineal')
    st.legend()
    st.show()


    #Mostrar el precio predecido para el dato dado
    daytest = [[196]]
    st.write('El modelo SVR RBF predijo: ', rbf_svr.predict(daytest))
    st.write('El modelo SVR Lineal predijo: ', lin_svr.predict(daytest))
    st.write('El modelo SVR Polinomial predijo: ', pol_svr.predict(daytest))


    # Mostrar el precio real para el dato dado
    st.write('El precio real es: ', df['Adj Close'][21])

