import numpy as np
import pandas as pd
import time
import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de acciones')
    user_input = st.text_input('Introducir TICKER de la empresa' , 'MSFT')
    st.title('Model - KNN')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 10, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    # continuar con su codigo
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    #setting index as date
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']

    #plot
        # Describiendo los datos
    st.subheader('Datos del 2021 de Diciembre') 
    st.write(df.head())
    st.subheader('Historial del precio de cierre') 
    st.write(df['Close'])

