import os
import tensorflow as tf
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
 
def creat_tf(imgpath):
 
    cwd = os.getcwd()
    classes = os.listdir(cwd + imgpath)
    
    writer = tf.python_io.TFRecordWriter("New 3 labels.tfrecords")
    
    for index, name in enumerate(classes):
        class_path = cwd + imgpath + name + "/"
        print(class_path)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((224, 224))
                
                img_raw = img.tobytes()             
                example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
                writer.write(example.SerializeToString())  
                print(img_name)
    writer.close()

def read_example():
 
    for serialized_example in tf.python_io.tf_record_iterator("6 labels.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
    
        label = example.features.feature['label'].int64_list.value
        
        print (label)

        
if __name__ == '__main__':
    imgpath = '/New 3 labels/'
    creat_tf(imgpath)
 #   read_example()


def read_and_decode(filename, batch_size):
    
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
    #img = tf.cast(img, tf.float32)#*(1. / 255)
    label = tf.cast(features['label'], tf.int64) 
    
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = batch_size,
                                                    num_threads = 64,
                                                    capacity=2000,
                                                    min_after_dequeue=1500,
                                                   )

    return img_batch, tf.reshape(label_batch, [batch_size])
 

tfrecords_file = '6 labels.tfrecords'
BATCH_SIZE = 20

image_batch, label_batch = read_and_decode(tfrecords_file, BATCH_SIZE)

with tf.Session() as sess:
    
    i = 0 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    
    try:
        while not coord.should_stop() and i<1:
            image, label = sess.run([image_batch, label_batch])
            for j in np.arange(BATCH_SIZE):
                print('label: %d' % label[j])
                plt.imshow(image[j,:,:,:])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()