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
import glob
import cv2
BATCH_SIZE = 100
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY=0.99#滑动平均衰减率
MODEL_SAVE_PATH="./model/"
MODEL_NAME="CosCode_model"

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
	
	#实例化saver
	saver = tf.train.Saver()
	#运行Session
	with tf.Session() as sess:
		#参数初始化
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		#训练时的存档点
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path) 
			#通过 checkpoint 文件定位到最新保存的模型，若文件存在，则加载最新的模型
		for i in range(STEPS):
			#####################x和y的喂入接口
			xs = 
			ys = 
			
			#reshape
			#####################
			_, loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
			if i % 200 == 0:
				print("After %d training, loss on training_bach is %g" %(step,loss_value))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)




'''''''''''''''''''''''''''''''3/10'''''''''''''''''''''''''
def main():
	backward()

if __name__=="__main__":
	main()


