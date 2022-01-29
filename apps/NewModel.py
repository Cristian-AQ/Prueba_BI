import numpy as np
import pandas as pd
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,plot_confusion_matrix,plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Predicción de acciones')
    user_input = st.text_input('Introducir TICKER de la empresa' , 'MSFT')
    st.title('Model - PROPHET')
    ticker = user_input
    period1 = int(time.mktime(datetime.datetime(2010, 1, 1, 23, 59).timetuple()))
    period2 = int(time.mktime(datetime.datetime(2022, 1, 10, 23, 59).timetuple()))
    interval = '1d'
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    data = pd.read_csv(query_string)

    st.title('Stock Forecast App')
    n_years = st.slider('Years of prediction:', 1, 12)
    period = n_years * 365

    st.subheader('DATA')
    st.write(data.tail())

    # Plot raw data
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
 
    # Predict forecast with Prophet.
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
    
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)