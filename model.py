from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import feature_column
# from keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, models

import os
import pandas as pd

label = pd.DataFrame()
file_path="export_dataframe.csv"
data=pd.read_csv(file_path)

train, test = train_test_split(data, test_size=0.99)
train, val = train_test_split(train, test_size=0.4)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def df_to_dataset(dataframe, shuffle=True):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Label')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  return ds


train_ds = df_to_dataset(train) 
val_ds = df_to_dataset(val, shuffle=False) 
test_ds = df_to_dataset(test, shuffle=False) 

# print(train_ds)

for feature_batch, label_batch in train_ds.take(1):
#   print('Every feature:', list(feature_batch.keys()))
  feature_columns = []
  for feature in feature_batch.keys():
    #   feature_column = feature_batch[feature]
      feature_columns.append(feature_column.numeric_column(feature))
  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# label['Label']=data['label']
# data=data.drop('label',1)

# x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

# print(x_train.head())
# print(y_train.head())
# print(x_test.head())
# print(y_test.head())

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

class_names = ['book', 'car', 'gift', 'movie', 'sell', 'total']

model = tf.keras.models.Sequential([feature_layer])


# model.add(tf.keras.layers.Conv2D(16, (12, 2), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((8, 8)))
# model.add(tf.keras.layers.Conv2D(8, (6, 2), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((8, 8)))
# model.add(tf.keras.layers.Conv2D(64, (2, 5), activation='relu'))

model.add(tf.keras.layers.Dense(units=15, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=6, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_ds, batch_size=32, validation_data=val_ds, epochs=2)

# test_loss, test_acc = model.evaluate(test_ds)

# print(test_acc)
print(history.accuracy, "   Accuracy of train")
print(model.summary())

