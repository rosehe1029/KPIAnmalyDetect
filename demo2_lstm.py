import pandas as pd
import numpy as np
from random import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as pyplot
from sklearn.metrics import classification_report
# 设置显示中文
pyplot.rcParams['font.sans-serif'] = ['KaiTi']
pyplot.rcParams['axes.unicode_minus'] = False
def create_model(input_length):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(input_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

data=pd.read_csv("data_normal.csv", parse_dates=True, index_col='_time')
data=data.sort_index(ascending=True) 
train,test=data.iloc[:int(len(data)*0.7)],data.iloc[int(len(data)*0.7):]
print(train.shape,test.shape)

X_train,y_train=train.iloc[:,:-1],train.iloc[:,-1]
X_test,y_test=test.iloc[:,:-1],test.iloc[:,-1]

X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape, y_train.shape)

model = create_model(len(X_train[0]))
hist = model.fit(X_train, y_train, batch_size=64, validation_split = 0.2, epochs=50, shuffle=False, verbose=1)

pyplot.plot(hist.history['loss'], label='loss')
#pyplot.plot(hist.history['accuracy'], label='acc')
#pyplot.plot(hist.history['val_accuracy'], label='val_acc')
pyplot.legend()
pyplot.show()
pyplot.close()

X_test= X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
y_pred=model.predict(X_test)
y_pred_= np.argmax(y_pred, axis=1) 
target_names=['class0','class1']
print(classification_report(y_test,y_pred_,target_names = target_names))


