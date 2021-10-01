import pandas as pd
import numpy as np
data=pd.DataFrame(np.random.randint(0,19,size=(3,4)),index=['1','2','3'],columns=["a","b","c","d"],dtype=np.float)

# 添加
# s1={'E':6,'w':}
# data=data.append(s1,ignore_index=True)
print(data)

# loc标签索引 添加行
# data.loc[4]=[1,2,3,4,5]
# 添加列
# data2=np.random.randint(0,10,size=(3,1))
# data['w']=data2

# 删除
data=data.drop(index="1",columns="a")
print(data)
data.b=[1,21]
print(data)