import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import tensorflow as tf
 
out_path='./model/'
with tf.Session() as sess:
    # Load .ckpt file
    ckpt_path ='./model/model.ckpt'
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')
    saver.restore(sess, ckpt_path)
 
    # Save as .pb file
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants( sess,  graph_def,  ['output'])
    with tf.gfile.GFile(out_path+'new model.pb', 'wb') as fid:
        serialized_graph = output_graph_def.SerializeToString()
        fid.write(serialized_graph)



 
#model_dir = './'
out_path='C:/Users/xwen2/Desktop/AryllaNets1/model v2 (GC 3 Labels )/'
model_name = 'model.pb'
 
def create_graph():
    with tf.gfile.FastGFile(os.path.join(out_path+  model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
 
create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')
