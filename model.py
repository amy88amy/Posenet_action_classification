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


# def df_to_dataset(dataframe, shuffle=True):
#   dataframe = dataframe.copy()
#   labels = dataframe.pop('Label')
#   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#   if shuffle:
#     ds = ds.shuffle(buffer_size=len(dataframe))
#   return ds


# train_ds = df_to_dataset(train) 
# val_ds = df_to_dataset(val, shuffle=False) 
# test_ds = df_to_dataset(test, shuffle=False) 

# print(train_ds)

# for feature_batch, label_batch in train_ds.take(1):
# #   print('Every feature:', list(feature_batch.keys()))
#   feature_columns = []
#   for feature in feature_batch.keys():
#     #   feature_column = feature_batch[feature]
#       feature_columns.append(feature_column.numeric_column(feature))
#   feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# label['Label']=data['label']
# data=data.drop('label',1)

labels = data.pop('Label')
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.05)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

# print(x_train.head())
# print(y_train.head())
# print(x_test.head())
# print(y_test.head())

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

class_names = ['book', 'car', 'gift', 'movie', 'sell', 'total']

model = tf.keras.models.Sequential() # [feature_layer])


# model.add(tf.keras.layers.Conv2D(16, (12, 2), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((8, 8)))
# model.add(tf.keras.layers.Conv2D(8, (6, 2), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((8, 8)))
# model.add(tf.keras.layers.Conv2D(64, (2, 5), activation='relu'))

model.add(tf.keras.layers.Dense(units=128, input_shape=(6,), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=96, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=6, activation='softmax'))

# epochs=256
# learning_rate = 0.6
# decay_rate = learning_rate / epochs
# momentum = 0.8

# sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Hello!")
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=16, verbose=1)
print("Hello2!")
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accc: ", test_acc)
# print(model.accuracy, "   Accuracy of train")
print(model.summary())

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

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, test_loss, 'b', label='Test Loss')
plt.title('Training and Test loss')
plt.legend()




plt.show()


