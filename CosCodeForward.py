#CosCode, a universal convolution neural network.
#coding:utf--8
#This is: CosCodeForward.py
#2019/3/7,20:31
#Programmed in HNU,School of Robotics
#--------------
import tensorflow as tf


INPUT_SIZE = 32 #输入数据的尺寸
OUTPUT_NODE = 5 #输出数据的结点
NUM_CHANNELS = 3 #图像通道
#------卷积层设定-------
CONV1_KERNEL_SIZE = 5 #卷积核边长
CONV1_KERNEL_NUM =  32#卷积核数量
CONV2_KERNEL_SIZE = 5
CONV2_KERNEL_NUM = 64
#------神经网络设定------
FC1_NODES_NUM = 512 #中间层第一层节点数
#------FORWARD------

def get_weight(shape, regularizer, name):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1),name=name)# ⬇正则化（的运行函数），加到loss里面一起训练（运行）
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	# 添加至正则化
	return w
def get_bias(shape,name):
	b = tf.Variable(tf.zeros(shape),name=name)
	return b
def conv2d(x,w):#实例化tf.nn.conv2d函数的实例化，有全零填充，步长为1（这里的w在conv2d函数中代表filter）
	return tf.nn.conv2d(x, w, strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):#池化核2x2，移动步长2x2，全零填充
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x, train, regularizer):
	#conv1_w指构建conv1层卷积的卷积核参数                        #⬇输入端维度   ⬇输出端维度
	conv1_w = get_weight([CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM],regularizer,'conv1_w')
	#conv1层的偏置                                             
	conv1_b = get_bias([CONV1_KERNEL_NUM],'conv1_b')           
	#把w和b设置好之后，经过conv2d函数，得到第一层初步的卷积结果     
	conv1=conv2d(x,conv1_w)
	#先过激活还是先过池化都可以
	relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))      
	pool1=max_pool_2x2(relu1)
	'''
	增加图片的通道数，使用一张3×3五通道的图像（对应的shape：[1，3，3，5]），用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，仍然是得到
	一张3×3的feature map，这就相当于每一个像素点，卷积核都与该像素点的每一个通道做卷积。
	'''                                                         #输入维度          #输出维度
	conv2_w = get_weight([CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer,'conv2_w')
	conv2_b = get_bias([CONV2_KERNEL_NUM],'conv2_b')
	conv2=conv2d(pool1,conv2_w)
	relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
	pool2=max_pool_2x2(relu2)
	
	'''进行对矩阵的拉直操作'''
	pool_shape=pool2.get_shape().as_list()#得到pool2的形状
	nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]#pool2的长[1]X宽[2]X深度[3]（第一个是batch，我们不需要这个值）
	reshaped=tf.reshape(pool2,[pool_shape[0],nodes])#把原先的矩阵形态转换成一共有batch行，每行有nodes个元素的数组
	
	'''喂入神经网络'''
	fc1_w=get_weight([nodes,FC1_NODES_NUM],regularizer,'fc1_w')
	fc1_b=get_bias([FC1_NODES_NUM],'fc1_b')
	fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
	if train: fc1=tf.nn.dropout(fc1,0.5)#训练时对fc1的中间层节点进行随机dropout,可以有等同于正则化的效果
	
	fc2_w=get_weight([FC1_NODES_NUM, OUTPUT_NODE],regularizer,'fc2_w')
	fc2_b=get_bias([OUTPUT_NODE],'fc2_b')
	y=tf.matmul(fc1,fc2_w)+fc2_b#输出层不过激活函数
	return y

''''''''''''''''''''''''''''''''''||'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
	

  




