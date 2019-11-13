
import os
import pandas as pd

label = pd.DataFrame(columns=['Label'])
data = pd.DataFrame()

root_dir = os.getcwd() + "/CSV"
lab = []
i = 0
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        path = os.path.join(subdir, file)
        l = path.split('/')

        if l[-2] not in lab:
            lab.append(l[-2])
            i += 1
        
        df = pd.read_csv(path)
        df=df.drop(df.columns.difference(['leftShoulder_x','leftShoulder_y','rightShoulder_x',
        'rightShoulder_y','leftElbow_x','leftElbow_y','rightElbow_x','rightElbow_y','leftWrist_x',
        'leftWrist_y','rightWrist_x','rightWrist_y',]), 1)
      
        for k in range(0, df.shape[0]):
            label.loc[len(label.index), 'Label'] = int(i)
        data = pd.concat([data, df])
        data.append(label)

data['label']=label['Label']

export_csv = data.to_csv (r'export_dataframe.csv', index = None, header=True)