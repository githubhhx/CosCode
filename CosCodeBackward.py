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
	global_step = tf.variable(0,trainable=False)
	














