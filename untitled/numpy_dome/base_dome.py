import numpy as np
# 数据的索引、切片、布尔索引、数据值的替换、数字的广播机制

# reshape、resize 区别是否改变原先的数组
# data=[1,3,3,4]
# data1=[1,3,4,4,2,4,2,8]
# data=np.array(data)
# data1=np.array(data1)
# # print(data.reshape((2,4)))
# data1.resize((2,4))
# data.resize((1,4))
# #可以见的resize()方法不返回对象
# print(data1)

# flatten、ravel方法 对象引用  flatten 返回是个拷贝 后续修改并不改变原先数值  ravel是个引用
# data2=data1.flatten()
# data2[3]=150
# print(data2)
# print(data1)

# vstack hstack concatenate 叠加
# print(np.vstack([data,data1]))
# print(np.vstack([data1,data]))
# print(np.concatenate([data1,data],axis=0))

#数据的切割
# hsplit vsplit split    def split(ary, indices_or_sections, axis=0)
# data=np.array([[0., 1., 2., 3.],
#        [4., 5., 6., 7.],
#        [8., 9., 10., 11.],
#        [12., 13., 14., 15.]],dtype=int)
# print(np.hsplit(data, (0,1)))
# data4=np.split()
# print(data4)

# #转置T与transpose()  dot()
# data=np.array([[0., 1., 2., 3.],
#        [4., 5., 6., 7.],
#        [8., 9., 10., 11.],
#        [12., 13., 14., 15.]],dtype=int)
# data1=data.T
# print(data.dot(data1))

# 不拷贝 深拷贝copy(栈\堆各自拷贝一份) 浅拷贝view(生成一个栈区 指向堆区还是同一块)  栈区速度快 容量小  堆区速度小 容量大
# a(栈区)=np.array(1)[堆区]
# data=np.array([[0., 1., 2., 3.],
#        [4., 5., 6., 7.],
#        [8., 9., 10., 11.],
#        [12., 13., 14., 15.]],dtype=int)
# data2=data.view()
# print(data2 is data)
# data[1][2]=200
# print(data2)


# savetxt loadtxt   一般操作csv文件  可以设置header 但不能存储三维以上的数据
# def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
#             footer='', comments='# ', encoding=None):
# def loadtxt(fname, dtype=float, comments='#', delimiter=None,
#             converters=None, skiprows=0, usecols=None, unpack=False,
#             ndmin=0, encoding='bytes', max_rows=None):

# np.save() np.savez()  np.load()  一般用来存储非文本类型的文件 扩展名为.npy 后者扩展名为.npz(经过压缩)   同一类型  可以存储多维数组比起savetxt的优势

# 3使用内置的csv  列表、字典的形式读取与写入
# data=dict({"chen":"cjaf","sadfa":"7"})
import  csv
# with open(r'E:\untitled\dome1\data1.csv','r') as da:
#     # reader=csv.reader(da)
#     reader=csv.DictReader(da)
#     # reader是迭代器
#     # next(reader)
#     # x是列表形式获取数据
#     # DictReader 不会包含标题那行数据
#     # 通过字典的方式读取
#     for x in reader:
#         values={"name":x['h']}
#         print(values)

# DictWriter writer两种写入的方式
# with open("writer1.csv",'w',encoding="utf-8",newline='') as wr:
#     header=["name","age","banji"]
#     values=[("xiaohcen","554",'船舶与海洋工程'),('xiaowang','25','中医药'),('xiaowu','25','工商银行')]
#     wr=csv.writer(wr)
#     wr.writerow(header)
#     wr.writerows(values)

# DictWriter 写入方式
# with open('dictwriter.csv','w',encoding='utf-8',newline="") as dictw:
#     header=["name","age"]
#     # 用字典存储数据  需要加上header参数
#     dictw=csv.DictWriter(dictw,header)
#     dictw.writeheader()
#     values=[{"name":"xiaochen","age":"12"}]
#     dictw.writerows(values)

# 读取文件遇到数据是空值 NAN(not a number)  INF(infinity) 代表无穷大 两者都是浮点类型
# 两种方式进行处理 删除 填充

 # 数据类型转换astype   两个NAN的值是不相等
# data=np.random.randint(0,13,size=(3,4)).astype(np.float)
# # data=data.astype(np.float)
# data[1,0]=np.NaN
# data[2,2]=np.NaN
# 删除NAN 数值会将多维数组变为一维数组 其中~为convert
# data=data[~np.isnan(data)]
# print(np.array(data).reshape(2,5))
# 删除NAN所在的行
# line=np.where(np.isnan(data))[0]#np.where获取nan所在的行和列 [0]获取行
# data2=np.delete(data,line,axis=0)
# print(line)

# 替换
# with open("convert.csv",'w',encoding="utf-8",newline='') as con:
#      values=np.random.randint(0,12,size=(4,9)).astype(np.float)
#      values[1][1]=np.nan
#      values[2][2]=np.nan
#      con=csv.writer(con)
#      con.writerows(values)
# #     在加载的时候 进行类型转换  不过还是会出现  not convet string to float 解决办法就是找到空的数据类型(string)数据所在的位置并替换为nan
# data=np.loadtxt("convert.csv",encoding="utf-8",dtype=np.str,delimiter=',')
# # data[data==""]=np.NAN
# data=data.astype(np.float)
# data[np.isnan(data)]=0
# print(data.sum())
# print(data.sum(axis=1))
#
# # 获取一列的数的总和
# # print(data.shape[1])
# data2=data.astype(np.float)
# for i in range(data2.shape[1]):
#      cols=data2[:,i] #获取一列所有的行的数据
#      # print(cols)
#      No_nan_cols=cols[~np.isnan(cols)]
#      cols_mean=No_nan_cols.mean()
#      # print(cols[np.isnan(cols)])
#      cols[np.isnan(cols)]=cols_mean
# print(data2)

# np.random.seed()   np.random.rand  np.random.randn(生成均值为0，标准差为1的正态分布的值)  np.random.randint
# np.random.choice  从数组中随机选取数据
# np.random.shuffle()  打乱数组

# data=np.random.seed(1)
# data=np.random.rand(2,3)
# data=np.random.randn()
# print(data)
# data=np.random.randint(0,13)
# print(np.random.choice(data,size=3))
# print(np.random.choice(10,size=(1,5)))
# data=np.arange(100)
# print(data)
# np.random.shuffle(data)
# print(data)

# axis知识点 最外面的括号 axis=0 依次往里加1
# sum() max() delete()
# np.random.seed(1)

# data=np.random.randint(1,32,size=(3,3))
# print(data)
# print(np.sum(data, axis=0))
# print(data.sum(axis=0),end="")
# print(data.sum(axis=1))
# print("====")
# print(data)
# print(data.max(axis=0))#16
# print(data.max(axis=1))#13 12 16
# print("=====")
# print(np.delete(data,0,axis=0))
# print(np.delete(data,1,axis=1))
# print(data)

# 大于等于等操作 异或操作
# np.greater()  np.greater_equal() np.less() np.less_equal() np.not_equal()
# np.logical_and()  np.logical_or()  np.logical_xor() p.logical_xor(x < 1, x > 3)、

# print(data)
# print(np.all(data==3))
# print(np.any(data==22))

# np.argsort() np.sort() -np.sort(-a)== indexsort=np.argsort(-a)  np.take(data,indexsort)
# print(np.argsort(data, axis=1))
# print(-np.sort(-data))
# print(data)
# print(np.sort(data,axis=0))

data=np.random.randint(0,13,size=(3,20))
def  get_mean(x):
     y=x[np.logical_and(x!=x.max(),x!=x.min())].mean()
     print(y)
     print("===")
# print(data)
# print("="*33)
# # np.apply_along_axis(lambda data:data[(data!=data.max())&(data.min())].mean(),axis=1,arr=data )
# np.apply_along_axis(get_mean,axis=1,arr=data)
# print(np.linspace(0, 1, 10))
print(np.unique(data))