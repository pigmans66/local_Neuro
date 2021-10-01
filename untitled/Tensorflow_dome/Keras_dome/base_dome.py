import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import csv
import pandas as pd
# print(tf.__version__)
# print(tf.keras.__version__)
# 模型的设置
model=tf.keras.Sequential()
model.add(layers.Dense(32,activation="relu"))
model.add(layers.Dense(32,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

# 稀疏性 参数为0 正则项
# 经验风险最小化 变成结构风险最小化
# KKT
# layers设置
layers.Dense(32,activation="sigmoid")
layers.Dense(32,activation=tf.sigmoid)
#s使用正交矩阵初始化内层权重
inittializar=tf.keras.initializers.Orthogonal()
print(inittializar)
layers.Dense(32,kernel_initializer=inittializar)
layers.Dense(32,kernel_initializer=tf.keras.initializers.glorot_normal)
layers.Dense(32,kernel_initializer=tf.keras.regularizers.l2(0.01))
layers.Dense(32,kernel_initializer=tf.keras.regularizers.l1(0.01))
