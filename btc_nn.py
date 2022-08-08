import pandas as pd
import numpy as np
import tensorflow as tf
import os, random
from tensorflow import keras
from scipy import stats

# Data CSV filepath
csv_path = "/home/icebear/Tradebot/btc.csv"
wb_path = "/home/icebear/Tradebot/btc_nn_wb"

# NN Settings
use_percentage = 0.7
epochs = 50
batch_size = 100
start_std = 0.1
learning_rate = 0.00001

def split_to_train_test_sets(df, labels, fraction):
    length = int(fraction*len(df))
    full_range = np.arange(0,len(df),1)
    random.shuffle(full_range)
    inds = full_range[:length]
    ninds = full_range[length:]
    train_x = df.iloc[inds]
    test_x = df.iloc[ninds]
    train_y = labels.iloc[inds]
    test_y = labels.iloc[ninds]
    return (train_x,train_y), (test_x,test_y)

def get_batch(x_data,y_data,batch_size):
    inds = np.random.randint(0,len(y_data),batch_size)
    return x_data.iloc[inds], y_data.iloc[inds]

def loss_function(logits,labels):
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
    return cross_entropy

def nn_model(x_input,W1,b1,W2,b2,W3,b3):
    x = tf.add(tf.matmul(tf.cast(x_input,tf.float32),W1),b1)
    x = tf.nn.relu(x)
    x = tf.add(tf.matmul(x,W2),b2)
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x, W3), b3)
    return logits

##########################################################
# Read in data as pd.Dataframe
df = pd.read_csv(csv_path,header=None)
# separate the labels from the features
labels = df.iloc[:,0]
labels = labels.astype(int)
# Drop the labels from the dataframe
df.drop(df.columns[0],axis=1,inplace=True)
# Check for any nans, for now, fill with a random value
df.fillna(np.random.rand(),inplace=True)
# Get the test/training sets
(train_x,train_y), (test_x,test_y) = split_to_train_test_sets(df,labels,use_percentage)
test_x = tf.Variable(test_x)

# Initialize the weights and biases.
W1 = tf.Variable(tf.random.normal([len(train_x.values[0]),300],stddev=start_std), name="W1")
b1 = tf.Variable(tf.random.normal([300]), name="b1")
W2 = tf.Variable(tf.random.normal([300,300],stddev=start_std), name="W2")
b2 = tf.Variable(tf.random.normal([300]), name="b2")
W3 = tf.Variable(tf.random.normal([300,2],stddev=start_std), name="W3")
b3 = tf.Variable(tf.random.normal([2]), name="b3")

# Set up the optimizer
optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

# Main Model
total_batch = int(len(train_y)/batch_size)
for epoch in range(epochs):
    avg_loss = 0
    # Training step
    for i in range(total_batch):
        batch_x, batch_y = get_batch(train_x,train_y,batch_size)
        batch_x = tf.Variable(batch_x)
        batch_y= tf.Variable(batch_y)
        batch_y = tf.one_hot(batch_y,2)
        with tf.GradientTape() as tape:
            logits = nn_model(batch_x,W1,b1,W2,b2,W3,b3)
            loss = loss_function(logits,batch_y)
            gradients = tape.gradient(loss, [W1,b1,W2,b2,W3,b3])
            optimizer.apply_gradients(zip(gradients,[W1,b1,W2,b2,W3,b3]))
            avg_loss += loss.numpy()/total_batch
    # Testing Step
    test_logits = nn_model(test_x,W1,b1,W2,b2,W3,b3)
    max_inds = tf.argmax(test_logits,axis=1)
    test_acc = np.sum(max_inds.numpy() == test_y) / len(test_y)
    print(f'EPOCH # {epoch + 1}: loss={round(avg_loss,4)}; accuracy: {round(test_acc,4)}')

# Save weights and biases
np.save(os.path.join(wb_path,"W1"),W1.numpy())
np.save(os.path.join(wb_path,"W2"),W2.numpy())
np.save(os.path.join(wb_path,"W3"),W3.numpy())
np.save(os.path.join(wb_path,"b1"),b1.numpy())
np.save(os.path.join(wb_path,"b2"),b2.numpy())
np.save(os.path.join(wb_path,"b3"),b3.numpy())
print("DONE!")