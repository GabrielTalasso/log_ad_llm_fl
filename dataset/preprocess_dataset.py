import os
import pandas as pd
import re

regex = [
        r"(?<=blk_)[-\d]+", # block_id
        r'\d+\.\d+\.\d+\.\d+',  # IP
        r"(/[-\w]+)+",  # file path
        #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]

def apply_regex(line):
    for r in regex:
        line = re.sub(r, '<*>', line)
    return line

dataset_path = '../.dataset/hdfs/train.csv'
data = pd.read_csv(dataset_path)

data['Content'] = data['Content'].apply(apply_regex)
data.to_csv(dataset_path, index=False)

dataset_path = '../.dataset/hdfs/test.csv'
data = pd.read_csv(dataset_path)

data['Content'] = data['Content'].apply(apply_regex)
data.to_csv(dataset_path, index=False)


