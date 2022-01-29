import numpy as np
import pandas as pd
import time
import datetime
import streamlit as st

from fastai.tabular.core import add_datepart
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV


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
    
    #setting index as date
    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']

    st.subheader('Datos del 2021 de Diciembre') 
    st.write(df.head())
    st.subheader('Historial del precio de cierre') 
    st.write(df['Close'])
    
    #creating dataframe with date and the target variable
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
        
    add_datepart(new_data, 'Date')
    new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

    #split into train and validation
    train = new_data[:987]
    valid = new_data[987:]

    x_train = train.drop('Close', axis=1)
    y_train = train['Close']
    x_valid = valid.drop('Close', axis=1)
    y_valid = valid['Close']

    #scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_valid_scaled = scaler.fit_transform(x_valid)
    x_valid = pd.DataFrame(x_valid_scaled)

    #using gridsearch to find the best parameter
    params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)

    #fit the model and make predictions
    model.fit(x_train,y_train)
    preds = model.predict(x_valid)
    st.write(preds)
    