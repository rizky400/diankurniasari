from flask import Flask, render_template,url_for,request
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
#import tensorflow as tf
import datetime as dt
import seaborn as sns
import talib as tlb
import talib
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from keras.models import Sequential
from keras import layers
app = Flask(__name__)

model_file = open('model.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route("/")
def template_test():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """ import data saham"""
    #preprocessing
    data = pd.read_csv('C:/Users/QQ/Desktop/bu_dian/data.csv')
    #calculate ma & rsi
    for n in [5,10,20,35,55]:
    
        #Creating the moving average indicator and divide by Adj close
        data['ma'+str(n)] = tlb.SMA(data['Today'].values, timeperiod = n)/data['Today']
    
        #Create RSI Indicator
        data['rsi'+str(n)] = tlb.RSI(data['Today'].values/data['Today'],timeperiod = n)
    #data cleansing
    data.dropna(inplace = True)
    data.drop(['Today', 'pct change'], axis = 1, inplace = True)
    #one hot encoding
    encoder = LabelEncoder()
    categorical_data = ["direction"]
    for kolom in categorical_data:
        data[kolom] = encoder.fit_transform(data[kolom])
    #labeling
    label = pd.get_dummies(data['direction'])
    label.columns = ['direction_' + str(x) for x in label.columns]
    data= pd.concat([data, label], axis = 1)
    #splitting data
    X = data.iloc[:, 0:11].values
    Y = data.iloc[:, 11:14].values
    #convert array
    X = np.asarray(X)
    Y = np.asarray(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 33)
    #scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #building model
    model = Sequential()
    model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    model.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #fitting model
    model1 = model.fit(X_train,Y_train,validation_data=(X_test,Y_test))
    #import data input
    var1 = request.form['High']
    var2 = request.form['Low']
    var3 = request.form['Open']
    var4 = request.form['Close']
    var5 = request.form['Adj Close']
    vara = np.asarray(var1)
    varb = np.asarray(var2)
    varc = np.asarray(var3)
    vard = np.asarray(var4)
    vare = np.asarray(var5)
    data = [[vara,varb,varc,vard,vare]]
    #predicting model
    predict = model.predict(X_test)
    y = accuracy_score(Y_test.argmax(axis=1), predict.argmax(axis=1))

    return render_template('result.html',hasil = y)
if __name__=='__main__':
     app.run(debug=True)