from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow as tf
# from keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, models

import os
import pandas as pd

label = pd.DataFrame()
file_path="export_dataframe.csv"
data=pd.read_csv(file_path)
label['Label']=data['label']
data=data.drop('label',1)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

print(x_train.head())
print(y_train.head())
print(x_test.head())
print(y_test.head())

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

class_names = ['book', 'car', 'gift', 'movie', 'sell', 'total']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (17, 2), (2, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((4, 2)))
model.add(tf.keras.layers.Conv2D(32, (6, 2), (2, 1), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (2, 5), (2, 1), activation='relu'))
# model.add(Flatten())
model.add(Dense(units=34, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(test_acc)
print(model.summary())

