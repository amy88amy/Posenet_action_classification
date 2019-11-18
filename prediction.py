import pickle
import pandas as pd
import numpy as np 
import json
from pandas.io.json import json_normalize
from collections import defaultdict
import csv

def prediction(json_obj):
    json_list = json.loads(json_obj)
    df = pd.DataFrame.from_dict(json_list)
    df=df.sort_values('score',ascending=False)
    df=df.head(int(len(df)*0.9))
    df=df.drop('score',1)
    df_x=pd.DataFrame()
    df_y=pd.DataFrame()
    for value in df['keypoints']:
        
        df = json_normalize(value)
      
        df = df.drop(columns='score')
        
        df = df.T
        df = df.drop([1, 2, 3, 4, 11, 12, 13, 14, 15, 16], 1)
        df_x=pd.concat([df_x,df.iloc[[1]]])
        df_y=pd.concat([df_y,df.iloc[[2]]])

    dist=pd.DataFrame()
    print(df.columns, "    Collll")

    j=0
    df_x.columns = ['nose_x','leftShoulder_x','rightShoulder_x','leftElbow_x','rightElbow_x','leftWrist_x','rightWrist_x']
    df_y.columns = ['nose_y','leftShoulder_y','rightShoulder_y','leftElbow_y','rightElbow_y','leftWrist_y','rightWrist_y']
    df_x=df_x.reset_index(drop=True)
    df_y=df_y.reset_index(drop=True)
    for i in range(1,len(df_x.columns)):
        dist[df_x.columns[i]]=df_x[df_x.columns[0]]-df_x[df_x.columns[i]]
        dist[df_y.columns[i]]=df_y[df_y.columns[0]]-df_y[df_y.columns[i]]
    

    dist_x=dist[dist.columns[0]]-dist[dist.columns[2]]
    dist_y=dist[dist.columns[1]]-dist[dist.columns[3]]

    for i in range(0,len(dist.columns)-1,2):
        dist[dist.columns[i]]=dist[dist.columns[i]]-dist_x
        dist[dist.columns[i+1]]=dist[dist.columns[i+1]]-dist_y

    for col in dist.columns:
        dist[col] = dist[col].astype(float)

    
    class_names = ['book', 'car', 'gift', 'movie', 'sell', 'total']

    model1=pickle.load(open('XGBClassifier.sav','rb'))
    pred1=model1.predict(dist)
    d = defaultdict(int)
    for i in pred1:
        d[i] += 1
    label1i = max(d.iteritems(), key=lambda x: x[1])
    label1 = class_names[int(label1i[0])]

    print(label1)

    model2=pickle.load(open('GradientBoostingClassifier.sav','rb'))
    pred2=model2.predict(dist)
    d = defaultdict(int)
    for i in pred2:
        d[i] += 1
    label1i = max(d.iteritems(), key=lambda x: x[1])
    label2 = class_names[int(label1i[0])]

    print(label2)

    model3=pickle.load(open('KNN_model.sav','rb'))
    pred3=model3.predict(dist)
    d = defaultdict(int)
    for i in pred3:
        d[i] += 1
    label1i = max(d.iteritems(), key=lambda x: x[1])
    label3 = class_names[int(label1i[0])]

    print(label3)

    model4=pickle.load(open('Randomforest_model.sav','rb'))
    pred4=model4.predict(dist)
    d = defaultdict(int)
    for i in pred4:
        d[i] += 1
    label1i = max(d.iteritems(), key=lambda x: x[1])
    label4 = class_names[int(label1i[0])]

    print(label4)

    labels = [label1, label2, label3, label4]

#     json = "{
# “1”: “predicted_label”,
# “2”: “predicted_label”,
# “3”: “predicted_label”,
# “4”: “predicted_label”
# }

    print(labels)
   