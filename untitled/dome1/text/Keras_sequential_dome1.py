import tensorflow as tf
from tensorflow.keras import Sequential,layers,utils
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
from tensorflow.keras import Sequential,layers,utils
import matplotlib.pyplot as plt
import warnings
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
warnings.filterwarnings('ignore')
plt.rcParams["axes.unicode_minus"]=False
# datesets=pd.read_csv(r'E:\untitled\data_files\moment1.csv',usecols=[0,3],names=["time","Z"],index_col="time")
# datesets=datesets[0:91206:3]
# datesets.to_csv(r"E:\untitled\data_files\modify_moment.csv",encoding="UTF-8")
# print(datesets)
# plt.xlabel("time")
# plt.ylabel("Z moment")
# plt.plot(datesets)
# plt.savefig("figure1")
# plt.show()
datesets=pd.read_csv(r"E:\untitled\data_files\modify_moment.csv")
# plt.xlabel("time")
# plt.ylabel("Z moment")
# plt.plot(datesets["time"],datesets["Z"])
# plt.show()
# print(datesets.describe())
# print(datesets.shape)
# print(datesets)
# print(datesets.dtypes)
# #数据类型转换
# datesets['time']=pd.to_datetime(datesets['time'],format="%Y-%m-%d %H:%M:%S")
# print(datesets.dtypes)
# print(datesets['time'])
# datesets['time']=pd.to_datetime(datesets['time'])
# print(datesets.dtypes)
# datesets.drop(columns=['time'],axis=1,inplace=True)
# datesets=datesets.drop(columns=['time','Z'],axis=0,inplace=False)
# drop函数参数解释 columns要删除的列名可以多个列，axis 1表示纵向 0表示横向 inplace为true时表示 删除数据并替换原始数据 false 删除数据不替换源文件 放回一个新的数据集
# print(datesets)
# 删除1到2000行数据
datesets=datesets.drop(index=range(0,2000),axis=0)
datesets=datesets.drop(columns='time',axis=1)
print(datesets)
print("==================")
# 删除多行可以使用range函数，多列
# plt.plot(datesets['time'],datesets['Z'])
# plt.xlabel("time")
# plt.ylabel("Z Moment")
# plt.savefig('figure2')
# plt.show()
# 使用minmaxscaler()归一化 均值为0方差为1
# 缩放到0-1
# 最大最小标准化
print(datesets)
scaler1=MinMaxScaler()
datesets["Z"]=scaler1.fit_transform(datesets["Z"].values.reshape(-1,1))
# datesets["Z"].plot()
# plt.plot(datesets["time"],datesets["Z"],label="MinMaxScaler")
# plt.legend()
# plt.xlabel("time")
# plt.ylabel("Z Moment")
# plt.savefig("figure3")
# plt.show()
# print(datesets.dtypes)
# # datesets['Z']=scaler.fit_transform(datesets["Z"])
# print(datesets)
# plt.plot(datesets['time'],datesets['Z'])
# plt.xlabel("time")
# plt.ylabel("Z Moment")
# plt.savefig('figure3')
# plt.show()
# 缩放到-1到1
# 绝对值最大标准化
# scaler2=MaxAbsScaler()
# datesets["Z"]=scaler2.fit_transform(datesets["Z"].values.reshape(-1,1))
# print("======")
# max_abc=MaxAbsScaler()
# datesets["Z"]=max_abc.fit_transform(datesets["Z"].values.reshape(-1,1))
# print(datesets["Z"])
# plt.plot(datesets["time"],datesets["Z"],label="MaxAbsScaler")
# plt.legend()
# plt.xlabel("time")
# plt.ylabel("Z Moment")
# plt.savefig("figure4")
# plt.show()
# 标准归一化
# scalar3=StandardScaler()
# datesets["Z"]=scalar3.fit_transform(datesets["Z"].values.reshape(-1,1))
# plt.xlabel("time")
# plt.ylabel("Z Moment")
# plt.plot(datesets["time"],datesets["Z"],label='StandardScaler')
# plt.savefig("Standard_figure5")
# plt.legend()
# plt.show()
print("===================")
#特征工程  创建一个新的数据集 切分数据集进行交叉验证
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
    return np.array(x),np.array(y)
def split_dataset(x,y,train_ratio=0.8):
    x_len=len(x)
    train_data_len=int(x_len*train_ratio)
    x_train=x[:train_data_len]
    y_train=y[:train_data_len]
    x_text=x[train_data_len:]
    y_text=y[train_data_len:]
    return x_train,x_text,y_train,y_text
def create_batch_data(x,y,batch_size=32,data_type=1):
    if data_type==1:
        datesets=tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        test_batch_data=datesets.batch(batch_size)
        return test_batch_data
    else:
        datesets=tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
        train_batch_data=datesets.cache().shuffle(1000).batch(batch_size)
        return train_batch_data
#在这一步之前可以进行数据的整理 看异常数据 是否对模型的性能参生影响
datesets_original=datesets
print("原始数据集：",datesets_original)
SEQ_LEN=20
x,y=new_datesets(datesets_original.values,step_number=SEQ_LEN)
print(x.shape)
x_train,x_test,y_train,y_test=split_dataset(x,y,train_ratio=0.7)
print(x_train.shape)
test_batch_dataset=create_batch_data(x_test,y_test,batch_size=256,data_type=1)
train_batch_dataset=create_batch_data(x_train,y_train,batch_size=256,data_type=2)
print("测试训练集",train_batch_dataset)
model=Sequential([layers.LSTM(8,input_shape=(SEQ_LEN,1)),layers.Dense(1)])
print(utils.plot_model(model,"my_model.png",show_shapes=False))
# plot_model(model,to_file='name.png',show_shapes=True)
file_path="best_checkpoint.hdf5"
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=file_path,monitor="loss",save_best_only=True,save_weights_only=True)
# 优化器的选择
model.compile(optimizer='adam',loss="mae")
# 模型训练
history=model.fit(train_batch_dataset,epochs=10,validation_data=test_batch_dataset,callbacks=[checkpoint_callback])
# 可视化tensorbord
plt.figure(figsize=(16,8))
plt.plot(history.history['loss'],label='train_loss')
plt.plot(history.history['val_loss'],label='val loss')
plt.title("loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
test_pred=model.predict(x_test,verbose=1)
print(test_pred.shape,y_test.shape)
# 利用二方值来评估模型的性能
score=r2_score(y_test,test_pred)
print("r^2",score)
# 绘制模型验证结果
plt.figure(figsize=(16,8))
plt.plot(y_test,label="True label")
plt.plot(test_pred,label="Pred label")
plt.title("True VS Pred")
plt.savefig("True vs Pred")
plt.legend()
plt.show()
# 绘制test中前100个点的真值与预测值
y_true=y_test[1400:1500]
y_pred=test_pred[1400:1500]
fig,axes=plt.subplots(2,1,figsize=(16,8))
axes[0].plot(y_true,marker="o",color="red",linestyle="--",linewidth=1.0,label="y_true")
plt.legend()
axes[1].plot(y_pred,marker="*",color="blue",linestyle="--",label="y_pred")
plt.legend()
plt.savefig("1000number_date")
plt.show()
# 模型测试
# 预测一个样本
sample=x_test[-1]
sample=sample.reshape(1,sample.shape[0],1)
sample_pred=model.predict(sample)
ture_data=x_test[-1]
list(ture_data[:,0])
def predict_next(model,sample,epach=20):
    temp1=list(sample[:,0])
    for i in range(epach):
        sample=sample.reshape(1,SEQ_LEN,1)
        pred=model.predict(sample)
        value=pred.tolist()[0][0]
        temp1.append(value)
        sample=np.array(temp1[i+1:i+SEQ_LEN+1])
    return temp1
preds=predict_next(model,ture_data,100)
plt.figure(figsize=(16,8))
plt.plot(preds,color="yellow",label="Prediction")
plt.plot(ture_data,color="blue",label="Truth")
plt.xlabel("epach")
plt.ylabel("Value")
plt.legend(loc="best")
plt.show()