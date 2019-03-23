#CosCode, a universal convolution neural network.
#coding:utf--8
#This is: CosCodeBackward.py
#2019/3/7,20:31
#Programmed in HNU,School of Robotics
#--------------
import tensorflow as tf
import CosCodeForward
import os
import numpy as np


BATCH_SIZE = 30
REGULARIZER = 0.0001
STEPS = 5000
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率
MODEL_SAVE_PATH="./model/"
MODEL_NAME="CosCode_model"

'''数据导入/标签设置区'''
airplane_filename = []
airplane_label = []
automobile_filename = []
automobile_label = []
bird_filename = []
bird_label = []
cat_filename = []
cat_label = []
deer_filename = []
deer_label = []
filedir = r"C:\Users\10720\Desktop\cifar-10testpy\temptest"#训练集文件夹路径
'''-----------------'''
'''------get_filename函数根据不同训练集需要进行不同的修改------'''
def batch_generator(filedir):#一般先用的函数要先写在前面
	for filename in os.listdir(filedir +"/airplane"):
		airplane_filename.append(filedir+"/airplane/"+filename)
		airplane_label.append(0)
	for filename in os.listdir(filedir +"/automobile"):
		automobile_filename.append(filedir+"/automobile/"+filename)
		automobile_label.append(1)
	for filename in os.listdir(filedir +"/bird"):
		bird_filename.append(filedir+"/bird/"+filename)
		bird_label.append(2)
	for filename in os.listdir(filedir +"/cat"):
		cat_filename.append(filedir+"/cat/"+filename)
		cat_label.append(3)
	for filename in os.listdir(filedir +"/deer"):
		deer_filename.append(filedir+"/deer/"+filename)
		deer_label.append(4)
	filenamelist = np.hstack((airplane_filename,
						   automobile_filename,
						   bird_filename,
						   cat_filename,
						   deer_filename))
	labellist = np.hstack((airplane_label,
						automobile_label,
						bird_label,
						cat_label,
						deer_label))
	filenamelist = tf.cast(filenamelist, tf.string)
	labellist = tf.cast(labellist, tf.int32)
	input_queue = tf.train.slice_input_producer([filenamelist, labellist])
	
	image = tf.read_file(input_queue[0])
	label = input_queue[1]
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.resize_image_with_crop_or_pad(image, 32,32)
	#生成文件队列(批处理)
	image_batch, label_batch = tf.train.batch([image, label], batch_size=30, num_threads=1, capacity=10)
	return image_batch, label_batch
'''----------------------------------------------------'''
def backward():
	x = tf.placeholder(tf.float32,[      
			BATCH_SIZE,
			CosCodeForward.INPUT_SIZE,
			CosCodeForward.INPUT_SIZE,
			CosCodeForward.NUM_CHANNELS], name='x')
	y_ = tf.placeholder(tf.float32,[None,CosCodeForward.OUTPUT_NODE], name='y')
	y = CosCodeForward.forward(x,True,REGULARIZER)
	global_step = tf.Variable(0,trainable=False)
	'''关于损失函数'''
	'''分类任务用 交叉熵(cem)等'''
	'''回归任务用 均方误差(mse)                                  ⬇标签值为[a,b,c,d,...]中的最大值'''
	ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem=tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))#添加forward的正则化中的losses(运行正则化的函数)到loss中,一起运行（训练）。
	
	train_step=tf.train.AdamOptimizer(0.0001).minimize(loss,global_step=global_step)#选用adam优化器
	ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op=ema.apply(tf.trainable_variables())#apply所有的可训练的变量to ema，以维护一个影子变量（类似于模型的镜像，用于test）
	with tf.control_dependencies([train_step,ema_op]):
		train_op=tf.no_op(name='train')#把主要的“train_step”和另外的ema给绑定起来，这样就可以一下同时训练两个了
	
	''' '''
	xs_raw, ys_raw =  batch_generator(filedir)#xs,ys的队列必须在sess之前先定义好
	#实例化saver
	saver = tf.train.Saver()
	#运行Session
	with tf.Session() as sess:
		coord = tf.train.Coordinator()#开启线程读取图片，要不然读不了
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		
		
		#参数初始化
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		#训练时的存档点
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path) 
			#通过 checkpoint 文件定位到最新保存的模型，若文件存在，则加载最新的模型
		for i in range(STEPS):
			#####################xs和ys的喂入接口
			'''
			这里不能加xs_raw, ys_raw =  batch_generator(filedir)，否则就会重复生产队列，sess挂起。
			'''
			xs,ys=sess.run([xs_raw, ys_raw])
			ys = np.eye(5)[ys]#标签必须是one hot格式，即(1,0,0,0,0)这样的，可以前面占位符处使用tf.onehot,这里为了方便用np.eye
			#####################
			_, loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
			if i % 10 == 0:
				print("After %d training, loss on training_bach is %g" %(step,loss_value))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
		
		coord.request_stop()#关闭线程
		coord.join(threads)







'''
def PictureRead(filename):
	
	#包括图片和标签所以用slice_input_producer，若只读取图像可以用string_input_producer
	
	file_queue = tf.train.string_input_producer(file_list)#文件名序列
	reader = tf.WholeFileReader()#图片阅读器
	key,value = reader.read(file_queue)#key:filename;value:picture value(uint-8格式，tf看不懂的)
	image = tf.image.decode_jpeg(value)#把uint8解码成数字，好处理
	#图片修剪，定通道
	image_resized = tf.image.resize_images(image,[32,32])
	image_resized.set_shape(shape=[32,32,3])
	#批处理(image_batch是一个四维矩阵)【可能有用的信息（见代码最后）】
	image_batch = tf.train.batch([image_resized],batch_size=30,num_threads=1,capacity=10)
	
	with tf.Session() as sess:
		coord = tf.train.Coordinator()#开启线程读取图片，要不然读不了
		    # 只有调用 tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中，供计算单元调用，
			#否则会由于内存序列为空，数据流图会处于一直等待状态
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		
		key_,image_batched = sess.run([key,image_batch])
		
		coord.request_stop()
		coord.join(threads)
	
	return image_batched
	
'''	
	
'''''''''''''''''''''''''''''''3/10'''''''''''''''''''''''''
def main():

	backward()

if __name__=="__main__":
	main()
	
	
'''
num_threads：用来控制入队tensors线程的数量，如果num_threads大于1，则batch操作将是非确定性的，输出的batch可能会乱序
'''






