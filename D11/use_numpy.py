#%% md
# # 列表如何转为ndarray
#%%
#列表转ndarray
import numpy as np

list1 = [1, 2, 3, 4]
print(list1)
print(type(list1))


#%%
oneArray = np.array(list1)
print(type(oneArray))  #类型是ndarray
print(oneArray)
#%% md
# # ndarray与Python原生list运算效率对比
#%%
import random
import time
import numpy as np

#随机1亿个数据
a = []
for i in range(100000000):
    a.append(random.random())
print('随机完毕')
#%%
b = np.array(a)  # 转换为ndarray
print('转换完毕')
#%%
#ndarray的计算效率优于python的list
t1 = time.time()
sum1 = sum(a)
t2 = time.time()

t4 = time.time()
sum3 = np.sum(b)
t5 = time.time()
print(t2 - t1, t5 - t4)
#%%
t1 = np.array([1, 2, 3])
print(t1)
print(type(t1))
#%%
print(type(range(10)))  # range返回的是一个range对象，不能直接转换为ndarray
t2 = np.array(range(10))
print(t2)
print(type(t2))
#%%
t3 = np.arange(0, 10, 2)  # 等价于range(0,10,2),返回ndarray
print(t3)
print(type(t3))
#%%
#二维列表转ndarray
import numpy as np

list2 = [[1, 2], [3, 4], [5, 6]]

twoArray = np.array(list2)
print(type(twoArray))
print(twoArray)
print(list2)  #列表的输出是有逗号的，ndarray的输出是没有逗号的
#%%
twoArray.tolist()  # ndarray转换为列表
#%% md
# # 4.3 常用属性
#%%
#看ndarray的各种属性
list2 = [[1, 2], [3, 4], [5, 6]]

twoArray = np.array(list2)
# 获取数组的维度( 注意： 与函数的参数很像) 
print(twoArray.ndim)

# 形状（行，列） (0 axis,1 axis)
print(twoArray.shape)

# 有多少个元素
print(twoArray.size)
# 数据类型
print(twoArray.dtype)  #默认int32,4个字节，数据范围 -21亿到21亿
#%% md
# # 4.4 调整数组的形状
#%%
four = np.array([[1, 2, 3], [4, 5, 6]])

print(four)
# 修改的是原有的
four1 = four
print(id(four))
#%%
four.shape = (3, 2)  # 改变形状
print(id(four))
print(id(four1))
print(four)
#%%
print('-' * 50)
# 返回一个新的数组，reshape后id和之前的不一样
four2 = four.reshape(2, 3)
print(four)
print(id(four))
print(id(four2))
four2
#%%
# 将多维变成一维数组
five = four.reshape((6,), order='C')
# 默认情况下‘C’以行为主的顺序展开，‘F’（Fortran风格）意味着以列的顺序展开
six = four.flatten()  # 展开成一维数组
print(five)
print('-' * 50)
print(six)

# print(five.reshape(3,2))
#%%
seven = five.reshape(3, 2)  # 改变形状
seven
#%%
print('-' * 50)
# 拓展：数组的形状
t = np.arange(24)  #和列表的range函数一样，生成一个数组
print(t)
print(f'shape{t.shape}')
print(t.ndim)
# 转换成二维
t1 = t.reshape((4, 6))
print(t1)
print(t1.shape)
#%%
# 转成三维
#最前面的是零轴，轴越小，越靠外，最小的轴是最外层
# 0轴，1轴，2轴
t2 = t.reshape((2, 3, 4))
print(t2)
print(t2.shape)
print(t2.ndim)
#%%
t3 = t.reshape((2, 3, 2, 2))
print(t3)
print(t3.shape)
print(t3.ndim)
#%% md
# 4.5 数组转换为列表
#%%
#将数组转换为列表 tolist
a = np.array([9, 12, 88, 14, 25])
print(type(a))
list_a = a.tolist()
print(list_a)
print(type(list_a))

#%% md
# # 5 NumPy的数据类型
#%%
import random

f = np.array([1, 2, 3, 4, 127], dtype=np.int8)  # 返回数组中每个元素的字节单位长度,dtype设置数据类型
print(f.itemsize)  # 1 np.int16(一个字节)
# 获取数据类型
print(f.dtype)
#%%
f[4] = f[4] + 1  #溢出
#%%
f[4]
#%%
# 调整数据类型
f1 = f.astype(np.int64)
print(f1.dtype)
print(f1.itemsize)  #显示字节数
print('-' * 50)
#%%
random.random()  # 随机生成0-1之间的小数
#%%
# 拓展随机生成小数
# 使用python语法，保留两位
print(round(random.random(), 2))
# np.float16 半精度
arr = np.array([random.random() for i in range(10)])
print(arr)
print(arr.itemsize)
print(arr.dtype)
# 取小数点后两位
print(np.round(arr, 2))
#%% md
# # 数组和数的计算
#%%
# [1,2,3]+5
#%%
# 由于numpy的广播机机制在运算过程中，加减乘除的值被广播到所有的元素上面
t1 = np.arange(24).reshape((6, 4))
print(t1)
print("-" * 20)
t2 = t1.tolist()
print(t1 + 2)
# print(t2+2) 不能对列表进行直接加整数操作
print("-" * 20)
print(t1 * 2)
print("-" * 20)
print(t1 / 2)

#无论多少维的ndarray都可以直接和一个常数进行运算
#%% md
# ndarray与ndarray的运算
#%%
#形状相同,才可以进行加减乘除
t1 = np.arange(24).reshape((6, 4))
t2 = np.arange(100, 124).reshape((6, 4))
print(t1)
print(t2)
print('-' * 50)
print(t1 + t2)
print('-' * 50)
print(t1 * t2)  #是否是矩阵乘法？不是

#%%
#shape不同，不能进行运算
t1 = np.arange(24).reshape((4, 6))
t2 = np.arange(6).reshape((3, 6))
print(t1)
print(t2)
print(t1 - t2)
#%%
#一维数组和二维数组进行运算时，一维的元素个数和列数相等
t1 = np.arange(24).reshape((4, 6))
t2 = np.arange(6).reshape((1, 6))
print(t2.shape)
print(t1)
print(t2)
t1 - t2

#%%
(t1 - t2).shape
#%%
t1 = np.arange(24).reshape((4, 6))
t2 = np.arange(4).reshape((4, 1))
print(t2)
print(t1)
t1 - t2

#%%
#随机3维数组
a = np.arange(24).reshape((2, 3, 4))
b = np.arange(12).reshape((1, 3, 4))
a - b
#%% md
# 结论：ndim要相同，可以某一个轴的size不同，但是其中一个ndarray必须对应的轴的size为1
#%% md
# # 练习轴
#%%
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print("-" * 20)
print(np.sum(a, axis=0))  # [5 7 9]，按哪个轴求和，哪个轴没了
print("-" * 20)
print(np.sum(a, axis=1))  # [ 6 15]
print("-" * 20)
print(np.sum(a))
print("-" * 20)

#%%
a = np.arange(24).reshape((2, 3, 4))
b = np.sum(a, axis=0)
print("-" * 20)
print(b.shape)
c = np.sum(a, axis=1)
print("-" * 20)
print(c.shape)
d = np.sum(a, axis=2)
print("-" * 20)
print(d.shape)
#%% md
# # 索引和切片
#%%
import numpy as np

a = np.arange(10)
# 冒号分隔切片参数 start:stop:step 来进行切片操作print(a[2:7:2])# 从索引 2 开始到索引 7 停止，间隔为 2

# 如果只放置一个参数，如 [2]，将返回与该索引相对应的单个元素
print(a[0], a)

# 如果为 [2:]，表示从该索引开始以后的所有项都将被提取
print(a[2:])

print(a[2:8:2])  #切片是左闭右开

#%%
import numpy as np

t1 = np.arange(24).reshape(4, 6)
print(t1)
print('*' * 20)
print(t1[1])  # 取一行(一行代表是一条数据，索引也是从0开始的) print(t1[1,:]) # 取一行
print('*' * 20)
print(t1[1:])  # 取连续的多行
print('*' * 20)
print(t1[1:3, :])  # 取连续的多行
print('*' * 20)
print(t1[[0, 2, 3]])  # 取不连续的多行
#%%
#看列表
t2 = np.arange(24).reshape(4, 6).tolist()
t2[[0, 2, 3]]
#%%
print(t1[:, 1])  # 取一列
print('*' * 20)
print(t1[:, 1:])  # 连续的多列
print('*' * 20)
print(t1[:, [0, 2, 3]])  # 取不连续的多列
print('*' * 20)
print(t1[2, 3])  # # 取某一个值,三行四列  py是t1[2][3]
print('*' * 20)
print(t1[[0, 1, 1], [0, 1, 3]])  # 取多个不连续的值，[[行，行。。。],[列，列。。。]]

t1[1:3, 1:4]  #取1-3行，1-4列
#%% md
# # 修改值
#%%
import numpy as np

t = np.arange(24).reshape(4, 6)
print(t)
print(id(t))
print('-' * 50)
# # 修改某一行的值
# t[1,:]=0
#
# # 修改某一列的值
# t[:,1]=0
#
# # 修改连续多行
# t[1:3,:]=0
#
# # 修改连续多列
# t[:,1:4]=0
#
# # 修改多行多列，取第二行到第四行，第三列到第五列
# t[1:3,2:5]=0
#
# # 修改多个不相邻的点
# t[[0,1],[1,3]]=0

# 可以根据条件修改，比如讲小于10的值改掉
# t[t<10]=0

# 使用逻辑判断
# np.logical_and	& # np.logical_or	|
# np.logical_not		~
# t[(t>2)&(t<6)]=0	# 逻辑与，and
# t[(t<2)|(t>6)]=0	# 逻辑或，or
# t[~(t>6)]=0	# 逻辑非
# print(t)
t = t.clip(10, 18)
print(id(t))
t

#%%
t = np.arange(24).reshape(4, 6)
t < 10
#%%
#python的三目运算
a = 10
b = 15
c = a if a > b else b
c
#%%
# # 拓 展
# # 三目运算（ np.where(condition, x, y)满足条件(condition)，输出x，不满足输出y。)）
score = np.array([[80, 88], [82, 81], [75, 81]])
print(score)
result = np.where(score < 80, True, False)  #类似于if else
print(result)

#%%
score[result] = 100
score
#%% md
# # 数据的添加，删除与去重
#%%
# 1. numpy.append 函数在数组的末尾添加值。 追加操作会分配整个数组，并把原来的数组复制到新数组中。
# 此外，输入数组的维度必须匹配否则将生成ValueError。
'''
参 数 说 明 ：                                                                                                     arr： 输 入 数 组                                                                                                 values：要向arr添加的值，需要和arr形状相同（除了要添加的轴）                                                            axis：默认为 None。当axis无定义时，是横向加成，返回总是为一维数组！当axis有定义的时候，分别为0和1的时候。当
axis有定义的时候，分别为0和1的时候（列数要相同）。当axis为1时，数组是加在右边（行数要相同）。
'''
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print('第一个数组：')
print(a)
print('\n')

print(' 向 数 组 添 加 元 素 ：')
print(np.append(a, [7, 8, 9]))  #ndarray会被展平
print('\n')
print(a)
print('-' * 50)

#%%
b = np.array([[7, 8, 9]])
print(b.ndim)
print(b)
print('-' * 50)
#%%
print('沿轴 0 添加元素：')  #往哪个轴添加，那个轴size就变大
print(np.append(a, [[7, 8, 9]], axis=0))
print('\n')
print(a)
print('沿轴 1 添加元素：')
print(np.append(a, [[5, 5, 5], [7, 8, 9]], axis=1))
print(a)
print('-' * 50)

#%%
# # 2. numpy.insert
# # 函数在给定索引之前，沿给定轴在输入数组中插入值。# 如果值的类型转换为要插入，则它与输入数组不同。
# # 插入没有原地的，函数会返回一个新数组。 此外，如果未提供轴，则输入数组会被展开。

a = np.array([[1, 2], [3, 4], [5, 6]])

print('第一个数组：')
print(a)
print('\n')

print('未传递 Axis 参数。 在插入之前输入数组会被展开。')
print(np.insert(a, 1, [11, 22]))
print('-' * 50)
print(np.insert(a, 1, [11, 12], axis=0))  #那个轴插入，哪个轴size变大
# print('传递了 Axis 参数。 会广播值数组来配输入数组。')

print('沿轴  0 广播：')
print(np.insert(a, 1, 11, axis=0))
print('\n')

print('沿轴  1 广播：')
print(np.insert(a, 1, [1, 2, 5], axis=1))
#%%
#numpy.delete 函数返回从输入数组中删除指定子数组的新数组。 与 insert() 函数的情况一样，
# 如果未提供轴参数， 则输入数组将展开。
'''
参 数 说 明 ： arr： 输 入 数 组
obj：可以被切片，整数或者整数数组，表明要从输入数组删除的子数组axis：
沿着它删除给定子数组的轴，如果未提供，则输入数组会被展开'''
import numpy as np

a = np.arange(12).reshape(3, 4)

print('第一个数组：')
print(a)
print('\n')

print('未传递 Axis 参数。 在删除之前输入数组会被展开。')
print(np.delete(a, 5))
print('\n')
print(a)
print('删除第一行：')
print(np.delete(a, 1, axis=0))
print('\n')

print('删除第一列：')
print(np.delete(a, 1, axis=1))
#%%
# numpy.unique 函数用于去除数组中的重复元素。
'''
arr：输入数组，如果不是一维数组则会展平
return_index：如果为true，返回新列表元素在旧列表中的位置（下标），并以列表形式储
return_inverse：如果为true，返回旧列表元素在新列表中的位置（下标），并以列表形式储
return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数
'''
import numpy as np

a = np.array([5, 2, 6, 2, 7, 5, 6, 9, 8, 2])

print('第一个数组：')
print(a)
print('\n')

#去除重复元素，输出的是有序序列
# print ('第一个数组的去重值：')
# u = np.unique(a)
# print (u)
# print ('\n')

# print ('去重数组的索引数组：')
# u,indices = np.unique(a, return_index = True)
# print(u)
# print (indices)
# print ('\n')

# print ('我们可以看到每个和原数组下标对应的数值：')
# print ('去重数组的下标：')
# u,indices = np.unique(a,return_inverse = True)
# print (u)
# print (indices)
# print ('\n')

print('返回去重元素的重复数量：')
u, indices = np.unique(a, return_counts=True)
print(u)
print(indices)
u
#%% md
# # numpy的数学计算
#%%
import numpy as np

score = np.array([[80, 88], [82, 81], [75, 81]])
print(score)
# 1. 获取所有数据最大值
result = np.max(score)
print(result)
# 2. 获取某一个轴上的数据最大值，对应的轴就消失
result = np.max(score, axis=1)
print(result)
# 3. 获取最小值
result = np.min(score)
print(result)
# 4. 获取某一个轴上的数据最小值
result = np.min(score, axis=0)
print(result)
# 5. 数据的比较
result = np.maximum([-2, -1, 0, 1, 2], 0)  # 第一个参数中的每一个数与第二个参数比较返回大的
print(result)
result = np.minimum([-2, -1, 0, 1, 2], 0)  # 第一个参数中的每一个数与第二个参数比较返回小的
print(result)
result = np.maximum([-2, -1, 4, 1, 2], [1, 2, 3, 4, 5])
print(result)
# 接受的两个参数，也可以大小一致;第二个参数只是一个单独的值时，其实是用到了维度的广播机制；


'''通用函数：
numpy.sqrt(array)	平方根函数
numpy.exp(array)	e^array[i]的数组
numpy.abs/fabs(array)	计算绝对值
numpy.square(array)	计算各元素的平方 等于array	2
numpy.log/log10/log2(array)	计算各元素的各种对数
numpy.sign(array)	计算各元素正负号
numpy.isnan(array)	计算各元素是否为NaN
numpy.isinf(array)	计算各元素是否为NaN
numpy.cos/cosh/sin/sinh/tan/tanh(array) 三角函数
numpy.modf(array)	将array中值得整数和小数分离，作两个数组返回
numpy.ceil(array)	向上取整,也就是取比这个数大的整数
numpy.floor(array)	向下取整,也就是取比这个数小的整数
numpy.rint(array)	四舍五入
numpy.trunc(array)	向0取整
numpy.cos(array)	正弦值
numpy.sin(array)	余弦值
numpy.tan(array)	正切值

numpy.add(array1,array2)	元素级加法
numpy.subtract(array1,array2)	元素级减法
numpy.multiply(array1,array2)	元素级乘法
numpy.divide(array1,array2)	元素级除法 array1./array2
numpy.power(array1,array2)	元素级指数 array1.^array2
numpy.maximum/minimum(array1,aray2) 元素级最大值
numpy.fmax/fmin(array1,array2)	元素级最大值，忽略NaN
numpy.mod(array1,array2)	元素级求模
numpy.copysign(array1,array2)	将第二个数组中值得符号复制给第一个数组中值
numpy.greater/greater_equal/less/less_equal/equal/not_equal (array1,array2)
元素级比较运算，产生布尔数组
numpy.logical_end/logical_or/logic_xor(array1,array2)元素级的真值逻辑运算
'''


#%%
# 6. 求平均值
result = np.mean(score)  # 获取所有数据的平均值
print(result)
result = np.mean(score, axis=0)  # 获取某一行或者某一列的平均值
print(result)
# 7. 求前缀和
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)
print(arr.cumsum(axis=0))
'''
[1, 2, 3]------>	|1	|2	|3	|
[4, 5, 6]------>	|5=1+4 |7=2+5 |9=3+6|
'''
print(arr.cumsum(axis=1))
'''
[1, 2, 3]------>	|1	|2+1	|3+2+1	|
[4, 5, 6]------>	|4	|4+5	|4+5+6	|
'''



# 拓展：方差var, 协方差cov, 计算平均值 average, 计算中位数 median
#%%
# 8. argmin求最小值索引
result = np.argmin(score, axis=0)
res = np.min(score, axis=0)  #这样我们就可以知道最小的81是第二排的，是从前往后遍历的
print(result, res)

# 9. 求每一列的标准差（这里是总体标准差）
# 标准差是一组数据平均值分散程度的一种度量。一个较大的标准差，代表大部分数值和其平均值之间差异较大；
# 一个较小的标准差，代表这些数据较接近平均值反应出数据的波动稳定情况，越大表示波动越大，越不稳定。
result = np.std(score, axis=0)
print(result)

# 10. 极 值
result = np.ptp(score, axis=None)  #就是最大值和最小值的差
print(result)
#%%
t = np.arange(30).reshape(10, 3)
np.mean(t, axis=0)
#%% md
# # 数组的拼接
#%%
# 有的时候我们需要将两个数据加起来一起研究分析，我们就可以将其进行拼接然后分析
import numpy as np

# 1. 根据轴连接的数组序列，concatenate没有改变数组的维度
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[9, 10], [11, 12]])
# # 要求a,b两个数组的维度相同
print('沿轴 0 连接两个数组：')
print(np.concatenate((a, b), axis=0))  #写哪个轴，哪个轴就变化
print('-' * 50)

#%%
print('沿轴 1 连接两个数组：')
print(np.concatenate((a, b,c), axis=1))
#%%
#堆叠
print('-' * 50)
print(np.stack((a, b,c), axis=1).shape)  #stack会增加维度
#%%
# 2. 根据轴进行堆叠，2维会变为3维
arrays = [np.arange(12).reshape(3, 4) for _ in range(10)]
# print(arrays)
print(len(arrays))
print('沿轴 0 连接两个数组：')
result0 = np.stack(arrays, axis=0)
print(result0.shape)  #(10,3,4)
# print (result0)


#%%
print('-' * 50)
print('沿轴 1 连接两个数组：')
print(arrays[0])
result1 = np.stack(arrays, axis=1)
print(result1.shape)  #(3,10,4)
# print(result1)

#%%
print('-' * 50)
print('沿轴 2连接两个数组：这里-1和2是等价的')
result2 = np.stack(arrays, axis=-1)
print(arrays[0])
print(result2.shape)
# print(result2)
#%%


# # # 3. 矩阵垂直拼接，vstack没有增加维数，类似于concatenate   ---vertical
v1 = [[0, 1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10, 11]]
v2 = [[12, 13, 14, 15, 16, 17],
      [18, 19, 20, 21, 22, 23]]
result = np.vstack((v1, v2))
print(result)

# # # 4. 矩阵水平拼接，类似于concatenate  horizontal
v1 = [[0, 1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10, 11]]
v2 = [[12, 13, 14, 15, 16, 17],
      [18, 19, 20, 21, 22, 23]]
result = np.hstack((v1, v2))
print(result)
#%% md
# # 分割
#%%
# 1. 将一个数组分割为多个子数组
'''
参数说明：
ary：被分割的数组
indices_or_sections：是一个整数，就用该数平均切分，如果是一个数组，为沿轴切分的位置（左开右闭）
axis：沿着哪个维度进行切向，默认为0，横向切分。为1时，纵向切分
'''
import numpy as np

arr = np.arange(12).reshape(4, 3)
print(arr)
print('将数组分为三个大小相等的子数组：b是一个列表')
b = np.split(arr, 3, axis=1) #按那个轴，那个轴发生变化
print(b)
#%%
b[0].shape
#%%
b = np.split(arr, 4, axis=0)
print(b)
#%%
c = np.hsplit(arr, 3)
print(c)  #c是列表
print(c[0].shape)
#%%
d = np.random.random((2, 6))
print(d)
#%%
# # 2.numpy.hsplit 函数用于水平分割数组，通过指定要返回的相同形状的数组数量来拆分原数组。
harr = np.floor(10 * np.random.random((2, 6)))
print('原array：')
print(harr)

#%%
# # 3.numpy.vsplit 沿着垂直轴分割
a = np.arange(16).reshape(4, 4)

print('第一个数组：')
print(a)
print('\n')

print('竖直分割：')
b = np.vsplit(a, 2)
print(b)
#%% md
# # numpy inf和nan
#%%
'''
C 语言中表示最大的正整数值是 0x7FFF FFFF，最小的负整数是 0x8000 0000。
查阅资料后，发现 inf 表示无穷大，需要使用 float(‘inf’) 函数来转化，那么对应的就有	float('-inf') 表示无穷小了。
这样你就可以使用任意数来判断和它的关系了。
那什么时候会出现inf呢？ 比如一个数字除以0，Python中会报错，但是numpy中会是一个inf或者-inf
另外还有 nan，这种写法在 pandans 中常见，表示缺失的数据，所以一般用 nan 来表示。任何与其做运算结果都是 nan
'''
import numpy as np
#nan 是not a number的缩写，表示非数字，在numpy中表示一个非数字值，可以用来表示缺失数据
# 创建一个nan和inf #
a = np.nan
b = np.inf
print(a, type(a))
print(b, type(b))

#%%
# 并 且  np.nan != np.nan	结果 是TRUE
# 所以我们可以使用这两个结合使用判断nan的个数
print(np.nan == np.nan)
print(True == 1)
print(False == 0)
np.nan + 1  #nan和其他数运算的结果都是nan
#%%
# --判断数组中为nan的个数
t = np.arange(24, dtype=float).reshape(4, 6)

# 将三行四列的数改成nan
t[3, 4] = np.nan
t[2, 4] = np.nan
print(t)

#%%
# 可以使用np.count_nonzero() 来判断非零的个数
# print(np.count_nonzero(t))
print(t != t)
print('-' * 50)
print(np.count_nonzero(t != t))  # 统计nan的个数

#%%
# 将nan替换为0
t[np.isnan(t)] = 0
print(t)
#%%
# ----------练习： 处理数组中nan
t = np.arange(24).reshape(4, 6).astype('float')
#
# 将数组中的一部分替换nan
t[1, 3:] = np.nan
print(t)
print('-------------------')
print(t.shape)
print(id(t))
# 遍历每一列，然后判断每一列是否有nan
for i in range(t.shape[1]):
    #获取当前列数据
    temp_col = t[:, i]

    # 判断当前列的数据中是否含有nan
    nan_num = np.count_nonzero(temp_col != temp_col)
    # 条件成立说明含有nan
    if nan_num != 0:
        # 将这一列不为nan的数据拿出来,并计算平均值
        temp_col_not_nan = temp_col[temp_col == temp_col]
        print(temp_col_not_nan)
        # 将nan替换成这一列的平均值
        temp_col[np.isnan(temp_col)] = np.mean(temp_col_not_nan)

print(t)
print(id(t))
#%%
print(np.inf == np.inf)
np.inf
#%%
# np.nan和任何数据运算的结果都是nan
t = np.arange(24).reshape(4, 6).astype('float')
#
# 将数组中的一部分替换nan
t[1, 3:] = np.nan

t1 = np.arange(24).reshape(4, 6).astype('float')
t + t1
#%%
arr = np.array([-1, 0])
print(arr)
print(arr[0] / arr[1])  #1除0就会得到inf
#%%
np.nan + np.inf
#%% md
# # 转置和轴滚动
#%%
#对换数组的维度
import numpy as np

a = np.arange(12).reshape(3, 4)
print('原数组：')
print(a)
print('\n')

print('对换数组：')
print(np.transpose(a))
print(a)

# 与transpose一致
a = np.arange(12).reshape(3, 4)

print('原数组：')
print(a)
print('\n')

print('转置数组：')
print(a.T)
#%%
# 函数用于交换数组的两个轴
t1 = np.arange(24).reshape(4, 6)
re1 = t1.swapaxes(1, 0)

print(' 原 数 组 ：')
print(t1)
print('\n')
print(re1.shape)
print('调用 swapaxes 函数后的数组：')
print(re1)

#%%
t3 = np.arange(60).reshape(3, 4, 5)
print(t3.shape)
print('-' * 50)
t3 = np.swapaxes(t3, 1, 2)
print(t3.shape)
# print(t3) 数据不用记住，不用观察
#%%
# 数组的轴滚动,swapaxes每次只能交换两个轴，没有rollaxis方便，默认情况下轴滚动最前面
a = np.ones((3, 4, 5, 6))
# np.rollaxis(a, 2).shape
np.rollaxis(a, 3, 1).shape
#%%
#数据拷贝，copy()和赋值的区别
b = np.array([[1, 2, 3], [1, 2, 3]])
a = b.copy()
a
#%%
b[0, 0] = 3
print(b)
a
#%%
#随机数生成
arr = np.random.rand(2, 3, 4)
print(arr)
#%%
# 1. 均匀分布的随