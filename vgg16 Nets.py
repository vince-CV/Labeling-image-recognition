import tensorflow as tf
import numpy as np
#import cv2
from matplotlib import pyplot as plt
from tensorflow.python.framework import graph_util
import time

def read_and_decode(filename):
    
    filename_queue = tf.train.string_input_producer([filename])
 
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)   
    
    features = tf.parse_single_example(serialized_example, features={
                                           'label'   : tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)*(1./255)

    label = tf.cast(features['label'], tf.int64)

    return img, label

batch_size = 16
n_class = 6
learning_rate = 0.0001
max_step = 10000
#dropout_ratio = 0.8

training_images, labels = read_and_decode('./6 labels.tfrecords')
image_batch, label_batch = tf.train.shuffle_batch([training_images, labels], batch_size=batch_size, capacity=2000, min_after_dequeue=500)
label_batch = tf.one_hot(label_batch, n_class)


def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial)

def weight_variable_FT(shape, name):
    kernel = tf.constant(data_dict[name][0], name="weights")
    return kernel

def bias_variable_FT(shape, name):
    bias = tf.constant(data_dict[name][1], name="bias")
    return bias

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

def print_layer(t):
    print (t.op.name, ' ', t.get_shape().as_list(), '\n')

data_dict = np.load('./vgg16.npy', encoding='latin1').item()

def conv(x, d_out, name):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(data_dict[name][0], name="weights")
        bias = tf.Variable(data_dict[name][1], name="bias")
        
        conv = tf.nn.conv2d(x, kernel,[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        print_layer(activation)
        return activation


# Input-Layer
x =tf.placeholder(tf.float32, [None,224,224,3], name="input")
y_=tf.placeholder(tf.float32, [None,n_class], name="labels")

# Conv-Layers

conv1_1 = conv(x, 64, 'conv1_1')
conv1_2 = conv(conv1_1, 64, 'conv1_2')
pool1   = max_pool_2x2(conv1_2)
print_layer(pool1)
 
conv2_1 = conv(pool1, 128, 'conv2_1')
conv2_2 = conv(conv2_1, 128, 'conv2_2')
pool2   = max_pool_2x2(conv2_2)
print_layer(pool2)
 
conv3_1 = conv(pool2, 256, 'conv3_1')
conv3_2 = conv(conv3_1, 256, 'conv3_2')
conv3_3 = conv(conv3_2, 256, 'conv3_3')
pool3   = max_pool_2x2(conv3_3)
print_layer(pool3)
 
conv4_1 = conv(pool3, 512, 'conv4_1')
conv4_2 = conv(conv4_1, 512, 'conv4_2')
conv4_3 = conv(conv4_2, 512, 'conv4_3')
pool4   = max_pool_2x2(conv4_3)
print_layer(pool4)
 
conv5_1 = conv(pool4, 512, 'conv5_1')
conv5_2 = conv(conv5_1, 512, 'conv5_2')
conv5_3 = conv(conv5_2, 512, 'conv5_3')
pool5   = max_pool_2x2(conv5_3)
print_layer(pool5)

# FC-Layer

w_fc1=weight_variable([7*7*512, 4096])
b_fc1=bias_variable([4096])
h_pool2_flag=tf.reshape(pool5,[-1, 7*7*512])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flag,w_fc1)+b_fc1)
print_layer(h_fc1)
#dropout1 = tf.nn.dropout(h_fc1, dropout_ratio)

w_fc2=weight_variable([4096, 4096])
b_fc2=bias_variable([4096])
h_fc2=tf.nn.relu(tf.matmul(h_fc1,w_fc2)+b_fc2)
print_layer(h_fc2)
#dropout2 = tf.nn.dropout(h_fc2, dropout_ratio)

w_fc3=weight_variable([4096,n_class])
b_fc3=bias_variable([n_class])
y_conv=tf.nn.softmax(tf.matmul(h_fc2, w_fc3) + b_fc3, name="output")
print_layer(y_conv)


#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)),reduction_indices=[1]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

#train_step=tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accurace = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.global_variables_initializer().run()
    
    # Training
    for i in range(max_step):
        batch_x, batch_y = sess.run([image_batch, label_batch])
        if i%10 == 0:
            train_accurace=accurace.eval(feed_dict={x:batch_x, y_: batch_y})
            print("step %d,train accurace %g"%(i,train_accurace))
        train_step.run(feed_dict={x:batch_x, y_:batch_y})
    
    # Save model as .pb 
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile('expert-graph2.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

