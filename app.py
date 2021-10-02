from flask import Flask, render_template,url_for,request
import pandas as pd
import numpy as np
import pandas_datareader.data as web
#import matplotlib.pyplot as plt
#import tensorflow as tf
import datetime as dt
import seaborn as sns
import talib as tlb
import talib
#import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
#from keras import layers
app = Flask(__name__)

#model_file = open('model.pkl', 'rb')
#model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
