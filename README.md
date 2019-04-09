# Deep-Learning-Systems
Deep learning served for feature engineering stuff (experiments).

### For .pd file read and write. Identify the file types.


For this ID processing, tested on more images (random pictured) with 96.8% accuracy (2 failed on 63 images in total).
To transplant to mobile end, the potential way might be:
`1.` To use smaller structure;
`2.` To use lite frameworks on, such as `tensorflow lite`, `ncnn`, etc., which supports Android and IOS, and they all have C++ API.

Note: OpenCV 3 has introduced `cv::dnn` module, but after initial try there is still no luck to load a freezen tf graph, and throw the error as "Too much unspecific tensor..." or something. 

Solution: 
`1` Get rid of these tensor nodes such as dropout or batch-normalization related stuff, using data pre-processing to replace those functions of generalization;
`2` Introducing other framework API in end, caffe2 is a good try, some of them give the pre-trained MobileNet or YOLO stuff, and start with these model and transfer can be great tradeoff;
`3` Keep patient until opencv or tensorfolw rocks. 
