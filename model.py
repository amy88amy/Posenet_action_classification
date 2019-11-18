from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import feature_column
# from keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, models
from keras import optimizers

import os
import pandas as pd
import numpy as np
import pickle

label = pd.DataFrame()
file_path="export_dataframe.csv"
data=pd.read_csv(file_path)
data = data.sample(frac=1)
print(data.shape)

# data = data.head(1000)
# print(data.head)
# print(data.shape)

# train, test = train_test_split(data, test_size=0.3)
# train, val = train_test_split(train, test_size=0.3)

# print(len(train), 'train examples')
# print(len(val), 'validation examples')
# print(len(test), 'test examples')

labels = data.pop('Label')

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# print(x_train.head())
# print(y_train.head())
# print(x_test.head())
# print(y_test.head())

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

class_names = ['book', 'car', 'gift', 'movie', 'sell', 'total']

model = tf.keras.models.Sequential()


model.add(tf.keras.layers.Dense(units=512, input_shape=(12,), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=6, activation='softmax'))

# epochs=256
# learning_rate = 0.6
# decay_rate = learning_rate / epochs
# momentum = 0.8

# sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

adam = tf.keras.optimizers.Adam(learning_rate=1, decay=1/128)

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Hello!")
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=64, verbose=1)
print("Hello2!")
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accc: ", test_acc)
print("Test loss: ", test_loss)

print(model.summary())

filename = 'DNN.sav'
pickle.dump(model, open(filename, 'wb'))

print(history.history['acc'])

# metric=classifier.evaluate_generator(test_generator)
# classifier.evaluate_generator(test_generator,metrics=['acc','precision','recall','fmeasure'])
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()