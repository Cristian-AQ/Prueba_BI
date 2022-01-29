import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime
import streamlit as st

# from fastai.tabular.core import add_datepart
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import neighbors
# from sklearn.model_selection import GridSearchCV

# Machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
    
    df = df.dropna()
    df = df[['Open', 'High', 'Low', 'Close']]
    st.subheader('Datos del 2021 de Diciembre') 
    st.write(df.head())
    
    # Predictor variables
    df['Open-Close']= df.Open -df.Close
    df['High-Low']  = df.High - df.Low
    df =df.dropna()
    X= df[['Open-Close', 'High-Low']]
    st.write(X.head())
    
    # Target variable
    Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)
    
    # Splitting the dataset
    split_percentage = 0.7
    split = int(split_percentage*len(df))

    X_train = X[:split]
    Y_train = Y[:split]

    X_test = X[split:]
    Y_test = Y[split:]
    
    # Instantiate KNN learning model(k=15)
    knn = KNeighborsClassifier(n_neighbors=15)

    # fit the model
    knn.fit(X_train, Y_train)

    # Accuracy Score
    accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
    accuracy_test = accuracy_score(Y_test, knn.predict(X_test))
    
    st.subheader('Train_data Accuracy: %.2f' %accuracy_train)
    st.subheader('Test_data Accuracy: %.2f' %accuracy_test)

    # Predicted Signal
    df['Predicted_Signal'] = knn.predict(X)

    # SPY Cumulative Returns
    df['SPY_returns'] = np.log(df['Close']/df['Close'].shift(1))
    Cumulative_SPY_returns = df[split:]['SPY_returns'].cumsum() * 100

    # Cumulative Strategy Returns 
    df['Startegy_returns'] = df['SPY_returns']* df['Predicted_Signal'].shift(1)
    Cumulative_Strategy_returns = df[split:]['Startegy_returns'].cumsum() * 100

    # Plot the results to visualize the performance

    plt.figure(figsize=(10,5))
    plt.plot(Cumulative_SPY_returns, color='r',label = 'SPY Returns')
    plt.plot(Cumulative_Strategy_returns, color='g', label = 'Strategy Returns')
    plt.legend()
    plt.show()
    
    Std = Cumulative_Strategy_returns.std()
    Sharpe = (Cumulative_Strategy_returns-Cumulative_SPY_returns)/Std
    Sharpe = Sharpe.mean()
    st.subheader('Sharpe ratio: %.2f'%Sharpe)
    
    # #setting index as date
    # df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    # df.index = df['Date']

    # st.subheader('Datos del 2021 de Diciembre') 
    # st.write(df.head())
    # st.subheader('Historial del precio de cierre') 
    # st.write(df['Close'])
    
    # #creating dataframe with date and the target variable
    # data = df.sort_index(ascending=True, axis=0)
    # new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

    # for i in range(0,len(data)):
    #     new_data['Date'][i] = data['Date'][i]
    #     new_data['Close'][i] = data['Close'][i]
        
    # add_datepart(new_data, 'Date')
    # new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

    # #split into train and validation
    # train = new_data[:987]
    # valid = new_data[987:]

    # x_train = train.drop('Close', axis=1)
    # y_train = train['Close']
    # x_valid = valid.drop('Close', axis=1)
    # y_valid = valid['Close']

    # #scaling data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # x_train_scaled = scaler.fit_transform(x_train)
    # x_train = pd.DataFrame(x_train_scaled)
    # x_valid_scaled = scaler.fit_transform(x_valid)
    # x_valid = pd.DataFrame(x_valid_scaled)

    # #using gridsearch to find the best parameter
    # params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
    # knn = neighbors.KNeighborsRegressor()
    # model = GridSearchCV(knn, params, cv=5)

    # #fit the model and make predictions
    # model.fit(x_train,y_train)
    # preds = model.predict(x_valid)
    # #rmse
    # rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
    
    # st.subheader(f'RMSE: {rms}')
    
    # pd.options.mode.chained_assignment = None

    # #plot
    # valid['Predictions'] = 0
    # valid['Predictions'] = preds
    # # st.subheader('Datos del 2021 de Diciembre') 
    # st.write(valid[['Close', 'Predictions']])
    # st.write(train['Close'])
    
    