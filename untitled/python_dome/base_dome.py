# import  random
#
# # 随机生成各种颜色
# def random_color():
#     colorpoint=['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
#     color1=''
#     for i in range(6):
#         color1+=colorpoint[random.randint(0,14)]
#     return '#'+color1
# colorsample=random_color()
# print(colorsample)
#
# # format 字符串表达式的使用
# print("随机生成的颜色样本1{},样本2{}".format(colorsample,random_color()))
#
# # time 时间戳  当前时间
# import  time
# print(time.asctime())
#
# #zip 打包操作
# list1=[1,2,34]
# list2=["xiao","wu",'huang']
# listsample=zip(list1,list2)
# print(listsample)
#
#
# class Employee:
#     empCount = 0
#
#     def __init__(self, name, salary):
#         self.name = name
#         self.salary = salary
#         Employee.empCount += 1
#
#     def displayCount(self):
#         print()
#
#     def displayEmployee(self):
#         print()
#
#
# print(Employee.__doc__)
#

# 类的变量 类的属性增加


# import time
# # 类的属性的检查、删除、增加、获得  访问私有变量的通告对象._类名__私有对象
# class menber:
#     local_time=time.asctime()
#     __menber_level=0
#     _menber_victity=0
#
#     def __init__(self,name):
#         self.name=name
#     def walk(self):
#         print("{}行走的速度为{},目前等级为{}".format(self.name,self._menber_victity,self.__menber_level))
#     def run(self):
#         print(menber._menber_victity+1)
# member1=menber("小钱")
# print("该成员目前情况有")
# member1.walk()
# member1.run()
# print(member1._menber__menber_level)
# menber.age=15
#
# getattr(menber,"age")
# hasattr(menber,'age')
# setattr(menber,'age',10)
# delattr(menber,'age')

# 正则化表达
# import re
# line = "Catadsfafds dsfa da huuh Are smarter than dogs"
# matchObj = re.match(r'(.*) are ', line, re.M | re.I)
# if matchObj:
#     print("matchObj.group() : ", matchObj.group())
#     # print("matchObj.group(1) : ", matchObj.group(1))
#     # print("matchObj.group(2) : ", matchObj.group(2))
#     # print("matchObj.group(3) : ", matchObj.group(3))
#     # print("matchObj.group(4) : ", matchObj.group(4))
# else:
#     print("No match!!")

# import re
# line="Color was are odfa are dfa"
# line="Color was are odfa are dfa"
# # result1=re.match("(.*?) are",line)
# # result2=re.match("(.*) are",line)
# # result3=re.match(".* are",line)
# # result4=re.match(".*? are",line)
# # print(result1)
# # print(result2)
# # print(result3)
# # print(result4)
# result5=re.match("Color (.*?) are",line)
# result6=re.match("Color (.*) are",line)
# result7=re.match("Color .* are",line)
# result8=re.match("Color .*? are",line)
# print(result5)
# print(result6)
# print(result7)
# print(result8)
import re
print(re.search('www', 'www.runoob.com').span())
print(re.search('com', 'www.runoob.com').span())