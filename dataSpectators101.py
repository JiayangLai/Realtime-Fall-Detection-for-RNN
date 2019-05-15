# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:55:59 2019

@author: KYDNN
"""

import pandas as pd
#pd.read_csv('dataset/train/walk.csv').iloc[:50].plot(figsize=[22,5])
pathA = 'dataset_01/raw/WAL/WAL_1_1_annotated.csv'
pd.read_csv(pathA)[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].iloc[:1000].plot(figsize=[22,5])
pathB = 'dataset_01/kalman/WAL/WAL_1_1_annotated.csv'
pd.read_csv(pathB)[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].iloc[:4000].plot(figsize=[22,5])
pathC = 'dataset/kalman/WAL/WAL_1_1_annotated.csv'
pd.read_csv(pathC)[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].iloc[:4000].plot(figsize=[22,5])