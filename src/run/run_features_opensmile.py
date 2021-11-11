'''
Compute features from opensmile and save a csv file with results in data/csv/name_smile.csv
'''


import os
import sys
import glob
import opensmile
import numpy as np

destinity = "./data/csv/"
data_path = "../Saarbruecken_Voice_Database"
""" files = glob.glob(os.path.join(data_path, "*.wav"))
files.sort()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
    loglevel=2,
    logfile='smile.log',
)
# read wav files and extract emobase features on that file

feat = smile.process_files(files)
feat.to_csv('_smile.csv')
 """

i = 0
data = []
for r, d, f in os.walk(data_path):
    for file in f:
        if '.wav' in file:
            path = r + '/' + file
            print(str(i+1) + '. append: ' + file)
            data.append(path)
            i = i+1

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
    loglevel=2,
    logfile='smile.log',
)
# read wav files and extract emobase features on that file
print('Processing: ... ')
feat = smile.process_files(data)

print('Saving: ... ' + destinity+'_smile.csv')
feat.to_csv(destinity+'_smile.csv')
