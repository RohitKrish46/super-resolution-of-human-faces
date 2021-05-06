from __future__ import print_function

import cv2
import tensorflow as tf 
import numpy as np 
import os

def model(x, y, lr_size, scale, batch, lr ,d, s, m):
    """
    Implementation of FSRCNN: http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html.
    """
    
    channels = 1
    PS = channels * (scale*scale) # for sub-pixel, PS = Phase Shift
    bias_initializer = tf.constant_initializer(value=0.0)

    
    filters = [
        tf.Variable(tf.random_normal([5, 5, 1, d], stddev=0.1), name="f1"),           
        tf.Variable(tf.random_normal([1, 1, d, s], stddev=0.1), name="f2"),           
        tf.Variable(tf.random_normal([1, 1, s, d], stddev=0.1), name="f%d" % (3 + m)),
        tf.Variable(tf.random_normal([1, 1, d, PS], stddev=0.1), name="f%d" % (4 + m)) 
    ]
    bias = [
        tf.get_variable(shape=[d], initializer=bias_initializer, name="b1"),
        tf.get_variable(shape=[s], initializer=bias_initializer, name="b2"),
        tf.get_variable(shape=[d], initializer=bias_initializer, name="b%d" % (3 + m)),
        tf.get_variable(shape=[1], initializer=bias_initializer, name="b%d" % (4 + m))
    ]
    # add filters and biases for 'non-linear mapping' layers (depeding on m), and name them in order
    for i in range(0,m):
        filters.insert(i+2, tf.Variable(tf.random_normal([3, 3, s, s], stddev=0.1), name="f%d" % (3+i)))  
        bias.insert(i+2, tf.get_variable(shape=[s], initializer=bias_initializer, name="b%d" % (3+i)))

    #Model architecture
    
    x = tf.nn.conv2d(x, filters[0], [1, 1, 1, 1], padding='SAME', name="conv1")
    x = x + bias[0]
    x = prelu(x, "alpha1")
  
    
    x = tf.nn.conv2d(x, filters[1], [1, 1, 1, 1], padding='SAME', name="conv2")
    x = x + bias[1]
    x = prelu(x, "alpha2")
  
    
    for i in range(0,m):
        x = tf.nn.conv2d(x, filters[2+i], [1, 1, 1, 1], padding='SAME', name="conv%d" % (3+i))
        x = x + bias[2+i]
        x = prelu(x, "alpha{}".format(3+i))
  
    
    x = tf.nn.conv2d(x, filters[3+(m-1)], [1, 1, 1, 1], padding='SAME', name="conv%d" % (3+m))
    x = x + bias[3+(m-1)]
    x = prelu(x, "alpha{}".format(3+m))
    
    x = tf.nn.conv2d(x, filters[4+(m-1)], [1, 1, 1, 1], padding='SAME', name="conv%d" % (4+m))

    

    # sub-pixel
    x = tf.nn.depth_to_space(x, scale, data_format='NHWC')
    out = tf.nn.bias_add(x, bias[4+(m-1)], name = "NHWC_output")
  
    # -- --

    # some outputs
    out_nchw = tf.transpose(out, [0, 3, 1, 2], name="NCHW_output")
    psnr = tf.image.psnr(out, y, max_val=1.0)
    loss = tf.losses.mean_squared_error(out, y)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return out, loss, train_op, psnr

def prelu(_x, name):
    """
    Parametric ReLU.
    """
    alphas = tf.get_variable(name, _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.1),
                        dtype=tf.float32, trainable=True)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg
