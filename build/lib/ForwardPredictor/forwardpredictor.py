#!/usr/local/bin/python
#-*- coding: utf-8 -*-
from types import TracebackType
from typing import ForwardRef
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import math

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

class ForwardPredictor:

    STEPS = 5
    EPOCHS = 40
    batchSize = 2
    startDate = None
    endDate = None

    def __init__(self, steps=None, EPOCHS=None, batchSize=None):
        self.STEPS = steps if steps else 5
        self.EPOCHS = EPOCHS if EPOCHS else 40
        self.batchSize = batchSize if batchSize else 2

    def predict(self, values):
        originalValues = values
        values = np.array(values).astype('float32')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        values = values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
        scaled = scaler.fit_transform(values)
        reframed = self._series_to_supervised(scaled, self.STEPS, 1)
        values = reframed.values
        n_train_meses = 36 - (6+self.STEPS)
        train = values[:n_train_meses, :]
        test = values[n_train_meses:, :]
        x_train, y_train = train[:, :-1], train[:, -1]
        x_val, y_val = test[:, :-1], test[:, -1]
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))

        model = self._crear_modeloFF()
        history=model.fit(x_train,y_train,epochs=self.EPOCHS,validation_data=(x_val,y_val),batch_size=self.batchSize)
        results=model.predict(x_val)

        _values = np.array(originalValues).astype('float32')
        # values = np.log10(values)
        # normalize features
        _values = _values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
        _scaled = scaler.fit_transform(_values)
        _reframed = self._series_to_supervised(_scaled, self.STEPS, 1)
        _reframed.drop(_reframed.columns[[self.STEPS]], axis=1, inplace=True)
        _values = _reframed.values
        x_test = _values[6:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

        results=[]
        for i in range(2):
            parcial=model.predict(x_test)
            results.append(parcial[0])
            x_test=self._agregarNuevoValor(x_test,parcial[0])

        adimen = [x for x in results]    
        inverted = scaler.inverse_transform(adimen)
        predicted = pd.DataFrame(inverted)

        _returnOne = 0
        _returnTwo = 0
        try:
            _returnOne = predicted.values[0]
        except:
            pass

        try:
            _returnTwo = predicted.values[1]
        except:
            pass

        return _returnOne, _returnTwo


    def _agregarNuevoValor(self, x_test,nuevoValor):
        for i in range(x_test.shape[2]-1):
            x_test[0][0][i] = x_test[0][0][i+1]
        x_test[0][0][x_test.shape[2]-1]=nuevoValor
        return x_test


    def _crear_modeloFF(self):
        model = Sequential() 
        model.add(Dense(self.STEPS, input_shape=(1,self.STEPS),activation='tanh'))
        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
        return model

    

    def _series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg