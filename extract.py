# -*- coding:utf-8 -*-
import os
import cv2
import pandas as pd
import numpy as np
import configparser as cp
import matplotlib.pyplot as plt

RAW_DATA_PATH = 'E:\\my_python\\Realtime-Fall-Detection-for-RNN-master\\dataset\\kalman\\'

Label = {'STD': 1, 'WAL': 2, 'JOG': 3, 'JUM': 4, 'STU': 5, 'STN': 6, 'SCH': 7, 'SIT': 8, 'CHU': 9,
         'LYI': 10, 'FOL': 0, 'FKL': 0, 'BSC': 0, 'SDL': 0, 'CSI': 15, 'CSO': 16}

def extract_data(data_file, sampling_frequency):
    """
    从mobileFall中提取数据，用于做实验测试
    :param data_file:  原始数据文件
    :param sampling_frequency: 原始数据采集频率
    :return:
    """
    data = pd.read_csv(data_file, index_col=0)
    data_size = len(data.label)
    for i in range(data_size):
        data.iat[i, 10] = Label[data.iloc[i, 10]]

    col_data = np.arange(0, data_size, int(sampling_frequency/50))
    extract_data = data.iloc[col_data, [1, 2, 3, 4, 5, 6, 10]]

    save_path = './dataset/raw/' + os.path.abspath(os.path.dirname(data_file)+os.path.sep+".").replace(RAW_DATA_PATH, '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = './dataset/raw/' + data_file.replace(RAW_DATA_PATH, '')
    extract_data.to_csv(save_path, index=0)

def find_all_data_and_extract(path):
    """
    递归的查找所有文件并进行转化
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print('路径存在问题：', path)
        return None

    for i in os.listdir(path):
        if os.path.isfile(path+"/"+i):
            if 'csv' in i:
                extract_data(path+"/"+i, 200)
        else:
            find_all_data_and_extract(path+"/"+i)



def main():
    find_all_data_and_extract(RAW_DATA_PATH)

if __name__ == '__main__':
    main()
    # if os.path.exists('./dataset/train/BSC_1_1_annotated.csv') == False:
    #     print('./dataset/train/BSC_1_1_annotated.csv', '文件不存在！')
    # data = pd.read_csv('./dataset/train/BSC_1_1_annotated.csv')
    #
    # #show_data(data)
    # data = kalman_filter(data)
    # data.to_csv('./dataset/train/BSC_1_1_annotated.csv', index=False)
    # #show_data(data)
    # # a = data.iloc[4:5,0]
    # # print(a)
    # data = pd.read_csv('./dataset/train/STU_1_1_annotated.csv')
    #
    # show_data(data)

