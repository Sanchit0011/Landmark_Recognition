# Import necessary modules
import pandas as pd
import random
import requests, urllib

# Data sampling to retrieve 2% images from top 100 most frequent classes

data = pd.read_csv("train.csv")
data = data[data['landmark_id'] != 'None']

class_counts = data['landmark_id'].value_counts()[0:100].to_frame()
data = data[data['landmark_id'].isin(class_counts.index)]

def random_sampling(init_data, percent_sample):
    result_data = pd.DataFrame(columns = init_data.columns) 
    landmarks = init_data['landmark_id'].unique()
    for landmark in landmarks:
        landmark_data = init_data[init_data['landmark_id'] == landmark]
        sample_rows = int(percent_sample * len(landmark_data))
        random_sample_df = landmark_data.sample(n=sample_rows, random_state=1)
        result_data = result_data.append(random_sample_df)
    return result_data
        
final_data = random_sampling(data, 0.020446)

def check_url(url):
    try:
        headers = {
            "Range": "bytes=0-10",
            "User-Agent": "MyTestAgent",
            "Accept": "*/*"
        }

        res = urllib.request.urlopen(url)
        return res.code in range(200, 209)
    except Exception:
        return False

url_valid = []   
for url in final_data['url']:
    print(check_url(url))
    url_valid.append(check_url(url))

final_data['url_valid'] = url_valid
final_data = final_data[final_data['url_valid'] == True]
final_data.to_csv('final_data.csv', index=False)