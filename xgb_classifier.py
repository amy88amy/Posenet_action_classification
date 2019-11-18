# Gradient Boosting 
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import optimizers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from xgboost import XGBClassifier

label = pd.DataFrame()
file_path="export_dataframe.csv"
data=pd.read_csv(file_path)
data = data.sample(frac=1)

labels = data.pop('Label')
class_names = ['book', 'car', 'gift', 'movie', 'sell', 'total']
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

eval_set = [(x_train,y_train), (x_test,y_test)]

clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.3, max_depth=10, verbose=2, n_iter_no_change=15)
history = clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
predictions = [np.round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


precision = precision_score(y_test, predictions, labels=[0, 1, 2, 3, 4, 5], average=None )
print("Precision: " , (precision * 100.0))


recall = recall_score(y_test, predictions, labels=[0, 1, 2, 3, 4, 5], average=None)
print("Recall: ", (recall * 100.0))

# filename = 'GradientBoostingClassifier.sav'
# pickle.dump(clf, open(filename, 'wb'))

# cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4, 5])

# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest')
# ax.figure.colorbar(im, ax=ax)
# # We want to show all ticks...
# ax.set(xticks=np.arange(cm.shape[1]),
#         yticks=np.arange(cm.shape[0]),
#         # ... and label them with the respective list entries
#         xticklabels=class_names, yticklabels=class_names,
#         title='Confusion matrix',
#         ylabel='True label',
#         xlabel='Predicted label')

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# fmt = 'd'
# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], fmt),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")
# fig.tight_layout()

# plt.show()


clf = XGBClassifier(n_estimators=300, learning_rate=0.3, max_depth=10)
# n_estimators = 100 (default)
# max_depth = 3 (default)

history2 = clf.fit(x_train,y_train,eval_metric=["merror", "mlogloss"], eval_set=eval_set,verbose=2)

y_pred = clf.predict(x_test)
predictions = [np.round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# filename = 'XGBClassifier.sav'
# pickle.dump(clf, open(filename, 'wb'))


precision = precision_score(y_test, predictions, labels=[0, 1, 2, 3, 4, 5], average=None)
print("Precision: ", precision * 100.0)


recall = recall_score(y_test, predictions, labels=[0, 1, 2, 3, 4, 5], average=None)
print("Recall: ", (recall * 100.0))

results = clf.evals_result()

# epochs = len(results['validation_0']['merror'])
# x_axis = range(0, epochs)
# # plot log loss
# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
# ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
# ax.legend()
# plt.ylabel('Log Loss')
# plt.title('XGBoost Log Loss')
# plt.show()
# # plot classification error
# fig, ax = plt.subplots()
# ax.plot(x_axis, results['validation_0']['merror'], label='Train')
# ax.plot(x_axis, results['validation_1']['merror'], label='Test')
# ax.legend()
# plt.ylabel('Classification Error')
# plt.title('XGBoost Classification Error')
# plt.show()
