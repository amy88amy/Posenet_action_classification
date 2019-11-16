
import os
import pandas as pd
import numpy as np
import math
data = pd.DataFrame()
label = pd.DataFrame(columns=['Label'])
root_dir = os.getcwd() + "/CSV"
lab = []
df_xx=pd.DataFrame()
df_yy=pd.DataFrame()
df_x=pd.DataFrame()
df_y=pd.DataFrame()
dist=pd.DataFrame()
i = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        path = os.path.join(subdir, file)
        l = path.split('/')
        
        if l[-2] not in lab:
            lab.append(l[-2])
            i += 1
        # print(l[-2])
        df = pd.read_csv(path)
        
        df_xx=df.drop(df.columns.difference(['leftShoulder_x','rightShoulder_x',
        'leftElbow_x','rightElbow_x','leftWrist_x','rightWrist_x','nose_x',]), 1)
        df_yy=df.drop(df.columns.difference(['leftShoulder_y','rightShoulder_y','leftElbow_y',
        'rightElbow_y','leftWrist_y','rightWrist_y','nose_y']), 1)
       
        for k in range(0, df.shape[0]):
            label.loc[len(label.index), 'Label'] = int(i)
        df_x = pd.concat([df_x, df_xx])
        df_y = pd.concat([df_y, df_yy])
        
   
df_x=df_x.reset_index(drop=True)
df_y=df_y.reset_index(drop=True)

for i in range(1,len(df_x.columns)):
    dist[df_x.columns[i]]=np.sqrt((df_x[df_x.columns[0]]-df_x[df_x.columns[i]])**2
                              +(df_y[df_y.columns[0]]-df_y[df_y.columns[i]])**2)

dist = dist.reset_index(drop=True)
label = label.reset_index(drop=True)
# print(dist.shape)
dist['Label']=label
# print(dist.shape)
dist.to_csv (r'export_dataframe.csv', index = None, header=True)
