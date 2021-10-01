import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
datesets=pd.read_csv(r"E:\untitled\data_files\modify_moment.csv")
datesets=datesets.drop(index=range(0,2000),axis=0)
# print(datesets)
# list 列表数据(可以为不同的数据类型)存放的地址(指针)，array存放的是相同的数据类型 元组()
# print(datesets.shape[0])
def new_datesets(datesets,step_number=10):
    x=[]
    y=[]
    start = 0
    end = datesets.shape[0] - step_number
    for i in range(start,end):
      sample=datesets[i:i+step_number]
      label=datesets[i+step_number]
      x.append(sample)
      y.append(label)
      return np.array(x),np.array(label)
# sample,label=new_datesets(datesets)
# print(sample)
# print("=====")
# print(label)