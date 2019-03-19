import tensorflow as tf 
import numpy as np 
from AryllaNets import *
import cv2
from datetime import datetime


batch_size = 16
lr = 0.0001
n_cls = 6
max_steps = 100
 
def read_and_decode(filename):
    
    filename_queue = tf.train.string_input_producer([filename])
 
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)   
    
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label'   : tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32)#*(1. / 255)

    label = tf.cast(features['label'], tf.int64)

    return img, label
 
def train():

    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='label')

    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, n_cls)
    probs = tf.nn.softmax(output, name = 'output')
 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probs, labels=y))

    #train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
 
    correct_prediction = tf.equal(tf.argmax(probs,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    images, labels = read_and_decode('./6 labels.tfrecords')

    img_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=2000, min_after_dequeue=500)
    label_batch = tf.one_hot(label_batch, n_cls)
 
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for i in range(max_steps):

            batch_x, batch_y = sess.run([img_batch, label_batch])
            _, loss_val = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y, keep_prob: 0.5})
            if i%10 == 0:
                train_arr = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1})
                print ("%s: Step [%d]  Loss : %f, training accuracy :  %g" % (datetime.now(), i, loss_val, train_arr))
     
            #if (i + 1) == max_steps/2:
                #checkpoint_path = os.path.join(FLAGS.train_dir, './model/model.ckpt')
                #saver.save(sess, './model/model.ckpt')

        coord.request_stop()
        coord.join(threads)
        saver.save(sess, './model/model.ckpt')

        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        with tf.gfile.FastGFile('./model/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

 
 
if __name__ == '__main__':
    train()