import sklearn
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
label = pd.DataFrame()
file_path="export_dataframe.csv"
data=pd.read_csv(file_path)
data=data.sample(frac=1)
label['Label']=data['Label']
data=data.drop('Label',1)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.0001)

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(x_train, y_train.values.ravel())
knn_predictions = knn_classifier.predict(x_test)
test_acc = accuracy_score(y_test, knn_predictions)
print('KNN Test accuracy: {:.5f}'.format(test_acc))
filename = 'KNN_model.sav'
pickle.dump(knn_classifier, open(filename, 'wb'))

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1)

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train, y_train.values.ravel())
rf_predictions = rf_classifier.predict(x_test)
test_acc = accuracy_score(y_test, rf_predictions)
print('Randomforest Test accuracy: {:.5f}'.format(test_acc))
filename = 'Randomforest_model.sav'
pickle.dump(rf_classifier, open(filename, 'wb'))
