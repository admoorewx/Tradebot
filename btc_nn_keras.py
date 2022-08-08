import pandas as pd
import numpy as np
import tensorflow as tf
import os, random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Data CSV filepath
csv_path = "/home/icebear/Tradebot/btc_rnn.csv"
save_path = "/home/icebear/Tradebot/btc_nn"

# NN Settings
use_percentage = 0.7
epochs = 10
batch_size = 1000
learning_rate = 0.00001

# Read in data as pd.Dataframe
df = pd.read_csv(csv_path,header=None)
# separate the labels from the features
labels = df.iloc[:,0]
labels = labels.astype(float)
# Drop the labels from the dataframe
df.drop(df.columns[0],axis=1,inplace=True)

# Split to train/test sets
train_x, test_x, train_y, test_y = train_test_split(df, labels, test_size=(1.0-use_percentage), shuffle=True)

# Need to re-shape the labels to prep for normalization
train_y = np.asarray(train_y).reshape(len(train_y),1)
test_y = np.asarray(test_y).reshape(len(test_y),1)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
train_x = scaler_x.fit_transform(train_x)
train_y = scaler_y.fit_transform(train_y)
text_x = scaler_x.fit_transform(test_x)
test_y = scaler_y.fit_transform(test_y)

# Re-shape the data to fit the LSTM
train_x = np.asarray(train_x)
train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
test_x = np.asarray(test_x)
test_x = test_x.reshape(test_x.shape[0],test_x.shape[1],1)

# Creating the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32,return_sequences=True,input_shape=(train_x.shape[1],1),dropout=0.35))
model.add(tf.keras.layers.LSTM(16,return_sequences=False,dropout=0.2))
model.add(tf.keras.layers.Dense(1,activation='relu'))
model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
# Fit/train the model
model.fit(train_x,train_y,epochs=epochs,batch_size=batch_size)

# Use the test dataset to run predictions and analyze output in detail
errors = []
yhat = model.predict(test_x)
for i,y in enumerate(yhat):
    # if len(errors) > 0:
    #     y = y[0] - np.mean(errors)
    # else:
    y = y[0]
    errors.append(test_y[i][0] - y)

print("Testing Stats")
print(f'Bias: {round(np.mean(errors),3)}')
print(f'Std.: {round(np.std(errors),3)}')
print(f'MAE: {round(np.mean(np.absolute(errors)),3)}')

# Plot some statistics
plt.figure()
plt.hist(errors,bins=100,align='mid')
plt.xlabel("Error")
plt.ylabel("Count")
plt.title(f'BTC-USD Prediction Error\nBias: {round(np.mean(errors),3)}; Std.: {round(np.std(errors),3)}; MAE: {round(np.mean(np.absolute(errors)),3)}')
plt.savefig("btc_errors.png")

# Save the model
model.save(save_path)




