import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

def data_load():
    # Load data set
    # 读取数据集
    merged_data = pd.read_csv("data.csv", index_col='_time')
    print(merged_data.head())
    #merged_data.plot()

    # Data pre-processing
    # Split the training and test sets 
    merged_data=merged_data.sort_index(ascending=True) 
    dataset_train,dataset_test=merged_data.iloc[:int(len(merged_data)*0.7)],merged_data.iloc[int(len(merged_data)*0.7):]

    #dataset_train.plot(figsize = (12,6))

    """
    Normalize data
    """
    scaler = preprocessing.MinMaxScaler() # 归一化

    X_train = pd.DataFrame(scaler.fit_transform(dataset_train), # Find the mean and standard deviation of X_train and apply them to X_train
                                  columns=dataset_train.columns,
                                  index=dataset_train.index)

    # Random shuffle training data
    X_train.sample(frac=1)

    X_test = pd.DataFrame(scaler.transform(dataset_test),
                                 columns=dataset_test.columns,
                                 index=dataset_test.index)
    return X_train,X_test


# Build AutoEncoding model
def AutoEncoder_build( X_train, act_func):
    tf.random.set_seed(10)

    # act_func = 'elu'

    # Input layer:
    model = tf.keras.Sequential()  # Sequential() is a container that describes the network structure of the neural network, sequentially processing the model

    # First hidden layer, connected to input vector X.
    model.add(tf.keras.layers.Dense(10, activation=act_func,  # activation function
                                    kernel_initializer='glorot_uniform',  # Weight initialization
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0),
                                    # Regularization to prevent overfitting
                                    input_shape=(X_train.shape[1],)
                                    )
              )

    model.add(tf.keras.layers.Dense(2, activation=act_func,
                                    kernel_initializer='glorot_uniform'))

    model.add(tf.keras.layers.Dense(10, activation=act_func,
                                    kernel_initializer='glorot_uniform'))

    model.add(tf.keras.layers.Dense(X_train.shape[1],
                                    kernel_initializer='glorot_uniform'))

    model.compile(loss='mse', optimizer='adam')  # 设置编译器

    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True)

    return model


def AutoEncoder_main(model, Epochs, BATCH_SIZE, validation_split):
    # Train model for 100 epochs, batch size of 10:
    #     Epochs=100
    #     BATCH_SIZE=10

    factor = 0.5
    X_train,X_test=data_load()
    X_train_noise = X_train + factor * np.random.normal(0,1,size=X_train.shape) # 设置噪声
    history = model.fit(np.array(X_train), np.array(X_train),
                        batch_size=BATCH_SIZE,
                        epochs=Epochs,
                        validation_split=validation_split,  # Training set ratio
                        # shuffle=True,
                        verbose=1)
    return history


def plot_AE_history(history):
    plt.plot(history.history['loss'],
                 'b',
                 label='Training loss')
    plt.plot(history.history['val_loss'],
                 'r',
                 label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.ylim([0,.1])
    plt.show()
    plt.close()

X_train,X_test=data_load()    
model=AutoEncoder_build( X_train, act_func='relu')
history=AutoEncoder_main(model=model, Epochs=100, BATCH_SIZE=32, validation_split=0.5)
plot_AE_history(history)

X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred,
                      columns=X_train.columns)
X_pred.index = X_train.index

scored = pd.DataFrame(index=X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
plt.figure()
sns.distplot(scored['Loss_mae'],
             bins = 10,
             kde= True,
            color = 'blue')
plt.xlim([0.0,.5])
plt.show()
plt.close()

X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred,
                      columns=X_test.columns)
X_pred.index = X_test.index


threshod = 0.3
scored = pd.DataFrame(index=X_test.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['Threshold'] = threshod
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()

X_pred_train = model.predict(np.array(X_train))
X_pred_train = pd.DataFrame(X_pred_train,
                      columns=X_train.columns)
X_pred_train.index = X_train.index

scored_train = pd.DataFrame(index=X_train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
scored_train['Threshold'] = threshod
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])
plt.show()
plt.close()