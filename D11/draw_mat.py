#%%
import matplotlib.pyplot as plt
#%%
x = range(1,8) # x轴的位置
y = [17, 17, 18, 15, 11, 11, 13]
# 传入x和y, 通过plot画折线图
plt.plot(x,y)
plt.show()
#%% md
# #### 7.2 设置折线图的颜色、透明度、线宽、线型
#%%
x = range(1,8) # x轴的位置
y = [17, 19,20, 15, 11, 13, 13]
# 传入x和y, 通过plot画折线图
plt.plot(x, y, color='red',alpha=0.5,linestyle='--',linewidth=5)
plt.show()
#%%
#折点样式
x = range(1,8) # x轴的位置
# 传入x和y, 通过plot画折线图
plt.plot(x, y, marker='v') 
plt.show()
#%%
# 7.4设置的图片的大小和保存
import random
x = range(2,26,2) # x轴的位置
y = [random.randint(15, 30) for i in x]
#设置画布
plt.figure(figsize=(20,8),dpi=80)
plt.plot(x,y)
# plt.savefig('t1.png') #保存图片
plt.savefig('t1.svg') #保存矢量图
#%%
# 7.5 绘制x轴和y轴的刻度
x = range(2,26,2) # x轴的位置
y = [random.randint(16,31) for i in x] 
plt.figure(figsize=(20,8),dpi=80)
# 构造x轴刻度标签
# 构造x轴刻度标签
x_ticks_label = [f"{i}:00" for i in x]  # 格式化x轴刻度标签
plt.xticks(x,x_ticks_label,rotation = 45) #rotation = 45 让字旋转45度

# 设置y轴的刻度标签
y_ticks_label = ["{}℃".format(i) for i in range(min(y),max(y)+1)]   # 格式化y轴刻度标签
print(y_ticks_label)
plt.yticks(range(min(y),max(y)+1),y_ticks_label)    # 设置y轴刻度标签

plt.plot(x,y)
plt.show()
#%%
# 7.6设置显示中文
x = range(0,120)
y = [random.randint(11,31) for i in range(120)]

plt.figure(figsize=(20,8),dpi=80) 
plt.plot(x,y)
from matplotlib import font_manager     # 加载中文字体
# 加载中文字体
my_font = font_manager.FontProperties(fname='C:\\Windows\\Fonts\\simsun.ttc',size=18)   # 设置字体
plt.xlabel('时间',rotation=45,fontproperties=my_font) # 设置x轴标签
plt.ylabel("次数",fontproperties=my_font) # 设置y轴标签

# 设置标题
plt.title('每分钟跳动次数',color='red',fontproperties=my_font)

plt.show()
#%%
# 7.7一图多线

y1 = [1, 0, 1, 1, 2, 4, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1, 1, 1]
y2 = [1, 0, 3, 1, 2, 2, 3, 4, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]

x = range( 11, 31 )  # # 设置图形
plt.figure( figsize=(20, 8), dpi=80 )  # 设置画布大小
plt.plot( x, y1, color='red', label='自己' )
plt.plot( x, y2, color='blue', label='同事' )
# 设置x轴刻度
xtick_labels = ['{}岁'.format( i ) for i in x]   # 设置x轴刻度标签
my_font = font_manager.FontProperties( fname='C:\\Windows\\Fonts\\simsun.ttc', size=18 )    # 设置字体
plt.xticks( x, xtick_labels, fontproperties=my_font, rotation=45 )  # 设置x轴刻度标签
# 绘制网格（网格也是可以设置线的样式)
# alpha=0.4 设置透明度
plt.grid( alpha=0.4 )

#  添加图例(注意：只有在这里需要添加prop参数是显示中文，其他的都用fontproperties) # 设置位置loc : upper left、 lower left、 center left、 upper center
plt.legend( prop=my_font, loc='upper left' )

# 展示
plt.show()
#%%
# 7.8拓展（一图多个坐标系子图）
import numpy as np

x  =  np.arange(1,99) #划分子图
fig,axes=plt.subplots(2,2)
ax1=axes[0,0]
ax2=axes[0,1] 
ax3=axes[1,0] 
ax4=axes[1,1]

# fig=plt.figure(figsize=(20,10),dpi=80) #作图1
ax1.plot(x, x) # 作 图 2 
ax2.plot(x, -x) #作图3
ax3.plot(x, x**2)
ax3.grid(color='r', linestyle='--', linewidth=1,alpha=0.3) #作图4
ax4.plot(x, np.log(x)) 
plt.show()
#%%
x = np.arange(1,100)
#新建figure对象
fig=plt.figure(figsize=(20,10),dpi=80) #新建子图1
ax1=fig.add_subplot(2,2,1)
ax1.plot(x, x)
#新建子图2
ax3=fig.add_subplot(2,2,2)  
ax3.plot(x, x**2)
ax3.grid(color='r', linestyle='--', linewidth=1,alpha=0.3)  
# 新 建 子 图 3
ax4=fig.add_subplot(2,2,3)
ax4.plot(x, np.log(x)) 
# 新 建 子 图 4
ax2=fig.add_subplot(2,2,4)
ax2.plot(x, -x) 
plt.show()
#%% md
# # 8.绘制散点图
#%%
y = [11, 17, 16, 11, 12, 11, 12, 6, 6, 7, 8, 9, 12, 15, 14, 17, 18, 21, 16, 17, 20, 14, 15, 15, 15, 19, 21, 22, 22, 22,
     23]
x = range( 1, 32 )

# 设置图形大小
plt.figure( figsize=(20, 8), dpi=80 )
# 使用scatter绘制散点图
plt.scatter( x, y, label='3月份' )    # 绘制散点图
# 调整x轴的刻度
my_font = font_manager.FontProperties( fname='C:\\Windows\\Fonts\\simsun.ttc', size=10 )     # 设置字体

_xticks_labels = ['3月{}日'.format( i ) for i in x]    # 设置x轴刻度标签

plt.xticks( x[::3], _xticks_labels[::3], fontproperties=my_font, rotation=45 )  
plt.xlabel( ' 日 期 ', fontproperties=my_font )   
plt.ylabel( '温度', fontproperties=my_font )  
plt.grid( alpha=0.4 )
# 图 例
plt.legend( prop=my_font )
plt.show()
#%% md
# # 9.绘制条形图
#%%
my_font =  font_manager.FontProperties(fname='C:\\Windows\\Fonts\\simsun.ttc',size=16)
a = ['流浪地球','大圣归来','飞驰人生','大黄蜂','熊出没·原始时代','奥本海默']
b = [38.13,19.85,14.89,11.36,6.47,5.93]

plt.figure(figsize=(20,8),dpi=80) # 设置画布大小

# 绘制条形图的方法
'''
width=0.3  条形的宽度
'''
rects = plt.bar(range(len(a)),b,width=0.3,color='r') # 绘制条形图
plt.xticks(range(len(a)),a,fontproperties=my_font,rotation=45) # 设置x轴的标签
for rect in rects:
    height = rect.get_height() # 获取条形的高度
    plt.text(rect.get_x()+rect.get_width()/2,height+0.5,'%.2f'%height,ha='center',va='bottom',fontproperties=my_font) # 设置条形的数值

plt.show()
#%% md
# # 10 直方图
#%%
time = [131,   98,    125,   131,   124,   139,   131, 117, 128, 108, 135, 138, 131, 102, 107, 114,
119,   128,   121,   142,   127,   130,   124, 101, 110, 116, 117, 110, 128, 128, 115,   99,
136,   126,   134,   95,    138,   117,   111,78, 132, 124, 113, 150, 110, 117,  86,    95, 144,
105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123, 86, 101,   99, 136,123,
117,   119,   105,   137, 123, 128, 125, 104, 109, 134, 125, 127,105, 120,  107,   129, 116,
108,   132,   103,   136, 118, 102, 120, 114,105, 115, 132, 145, 119, 121,  112,   139, 125,
138,   109,   132,   134,156, 106, 117, 127, 144, 139, 139, 119, 140,   83,    110,   102,123,
107,   143,   115,   136, 118, 139, 123, 112, 118, 125, 109, 119, 133,112,  114,   122, 109,
106,   123,   116,   131,   127, 115, 118, 112, 135,115,   146,   137,   116,   103,   144,   83,    123,
111,   110,   111,   100,   154,136, 100, 118, 119, 133,   134,   106,   129,   126,   110,   111,   109,
141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126,114, 140, 103,
130,   141, 117, 106, 114, 121, 114, 133, 137,    92,121,    112,   146,   97,    137, 105,  98,
117,   112,   81,    97, 139, 113,134, 106, 144, 110, 137,  137,   111,   104,   117, 100, 111,
101,   110,105, 129, 137, 112, 120, 113, 133, 112,    83,    94,    146,   133,   101,131, 116,
111,   84, 137, 115, 122, 106, 144, 109, 123, 116, 111,111, 133, 150]

# 创建画布


#%%
plt.figure(figsize=(20, 8), dpi=100) # 3）绘制直方图
print(max(time),min(time))
# 设置组距
distance = 2
# 计算组数
group_num = int((max(time) - min(time)) / distance) # 绘制直方图
plt.hist(time, bins=group_num)

# 修改x轴刻度显示
plt.xticks(range(min(time), max(time))[::2])
plt.yticks(range(0, 20, 1))

# 添加网格显示
plt.grid(linestyle="--", alpha=0.5) # 添加网格显示

#添 加 x,  y 轴 描 述 信 息
plt.xlabel("电影时长大小",fontproperties=my_font)
plt.ylabel("电影的数据量",fontproperties=my_font)
# 4）显示图像
plt.show()
#%% md
# # 11 饼图
#%%
import matplotlib
label_list = ["第一部分", "第二部分", "第三部分"]  # 各部分标签
size = [30,30,40]    # 各部分大小
color = ["red", "green", "blue"]   # 各部分颜色
explode = [0, 0.05, 0] # 各部分突出值
# 绘制饼图
plt.figure(figsize=(20, 8), dpi=100)
# plt.pie(size, labels=label_list, colors=color, explode=explode, autopct='%1.1f%%', shadow=True, startangle=140)
matplotlib.rcParams['font.sans-serif']=['SimHei']
patches, l_text, p_text = plt.pie(size,explode=explode, colors=color, labels=label_list,
                                  labeldistance=1.1, autopct="%1.1f%%", shadow=True, startangle=90, pctdistance=0.1)
plt.axis("equal")  # 设置横轴和纵轴大小相等，这样饼才是圆的
plt.legend()
plt.show()