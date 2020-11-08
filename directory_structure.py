# Import necessary modules
import pandas as pd
import numpy as np
import os
import requests, urllib
from sklearn.model_selection import StratifiedShuffleSplit
import os, shutil
import glob

# Move image files to train, validate and test folders

final_data = pd.read_csv('final_data.csv')
final_data = final_data.sort_values(by='landmark_id', ascending=True)
final_data = final_data.reset_index(drop=True)

urls = final_data['url']

for url in urls:
    img_id = final_data.loc[final_data['url']==url, 'id'].item()
    
    if os.path.exists('./train_images/'+str(img_id)+'.jpg'):
        continue
    else:
        urllib.request.urlretrieve(url, './train_images/'+str(img_id)+'.jpg')

X = final_data['id']
y = final_data['landmark_id']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_id, test_id in sss.split(X, y):
    X_train, X_tmp = X.iloc[train_id], X.iloc[test_id]
    y_train, y_tmp = y.iloc[train_id], y.iloc[test_id]

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

for train_id, test_id in sss1.split(X_tmp, y_tmp):
    X_valid, X_test = X_tmp.iloc[train_id], X_tmp.iloc[test_id]
    y_valid, y_test = y_tmp.iloc[train_id], y_tmp.iloc[test_id]

from_path = "./train_images/"
moveto_path_valid = "./valid_images/"
moveto_path_test = "./test_images/"

for f in X_valid.iloc[:]:
    src = from_path+f+'.jpg'
    dst = moveto_path_valid+f+'.jpg'
    shutil.move(src, dst)

for f in X_test.iloc[:]:
    src = from_path+f+'.jpg'
    dst = moveto_path_test+f+'.jpg'
    shutil.move(src, dst)