#装饰器原理  装饰器同样需要定义函数 通过wrapper来重写函数  func 参数传递可以使用*arg  返回值可以使用return  {:.4}保留小数点后四位 if elif else语句
import  time
# def display_time(func):
#     def wrapper(*args):
#         t1=time.time()
#         result=func(*args)
#         t2=time.time()
#         print(t2-t1)
#         return result0
#     return wrapper
#
# def primer(num):
#     if num<2:
#         return False
#     elif num==2:
#         return False
#     else:
#         for i in range(2,num):
#             if num%i==0:
#                 return False
#         return True
#
# @display_time
# def prinums(maxnum):
#     counts=0
#     for i in range(2,maxnum):
#         if primer(i):
#             counts=counts+1
#     return counts
# counts=prinums(10000)
# print(counts)
# def display_prime(level):
# #这一步参数传递
#     print("level".format(level))
#     def wrapper(func):
#         def inwrapper(*args):
#             t1=time.time()
#             result=func(*args)
#             t2=time.time()
#
#             print("run time{:.4}".format(t2-t1))
#             return result
#         return inwrapper
#     return wrapper
# def prime(num):
#     if num<2:
#         return False
#     elif num==2:
#         return False
#     else:
#         for i in range(2,num):
#             if num%i==0:
#                 return False
#         return True
# @display_prime(level=100)
# def count_prime(MaxNum):
#     counts=0
#     for i in range(2,MaxNum):
#         if prime(i):
#             counts=counts+1
#     return counts

# print(count_prime(500))
# 内部函数引用外部函数  闭包
# 装饰器 接受被装饰的函数作为参数  而且还要继续调用一次  被装饰函数的函数带参数 ：只需要在最内部函数传入参数即可    return func(x,y) func为被装饰函数
def func():
    a=1
    def func1(num):#闭包函数  私有化了变量(引用了外部变量)
        print(num+a)
    return func1
var=func()
var(3)
