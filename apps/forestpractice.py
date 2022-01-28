# -*- coding: utf-8 -*-
"""ForestPractice.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LGe1tVRZsFTPS9DXAIplCH962ykuvQBV
"""
import numpy as np
import pandas as pd
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import streamlit as st

def app():
    st.markdown('''
                <a href='https://www.youtube.com/watch?v=hoPvOIJvrb8'>HOME</a>
    ''', unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de tendencia de acciones')
    user_input = st.text_input('Introducir cotización bursátil' , 'MSFT')
    st.title('Model - RANDOMFORESTCLASSIFIER')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 10, 23, 59).timetuple()))
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

    # Uso del modelo
    modelo = RandomForestClassifier()
    modelo.fit(X_train,y_train)
    st.subheader('Score del modelo') 
    st.write(modelo.score(X_test,y_test))

    # Mejorando el score
    # while True:
    #   modelo = RandomForestClassifier()
    #   modelo.fit(X_train,y_train)
    #   if modelo.score(X_test,y_test)>0.689:
    #     break
    # st.subheader('Nuevo score del modelo') 
    # st.write(modelo.score(X_test,y_test))

    #Visualizaciones 
    pred_modelo = modelo.predict(X_test)
    st.subheader('Classification Report')
    classification_report(y_test,pred_modelo)
    st.subheader('Confusion Matrix')
    plot_confusion_matrix(modelo,X_test,y_test)
    st.pyplot()

    st.subheader('ROC')
    plot_roc_curve(modelo, X_test, y_test, alpha = 0.8)
    st.pyplot()