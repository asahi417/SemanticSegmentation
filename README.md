# DeepLab v3+
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
Tensorflow implementation of [DeepLab v3 +](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf).
  

## Get started

```
git clone https://github.com/asahi417/SemanticSegmentation
cd SemanticSegmentation
pip install .
```

## Train model on benchmark dataset
Here is brief instruction how you could get DeepLab model to be trained on benchmark dataset
([PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) can be selected).  

### Setup dataset 
Firstly, you have to download and convert data to TFRecord format by

```
python ./bin/build_tfrecord.py -d pascal
```      
 
Once you've done the process, `data` directory will be created.

### Train model
After producing TFRecord file, one can start training DeepLav by 

```
python ./bin/train.py -d pascal --crop_size 321 --aspp_depth 128 --backbone xception_41 -b 12 --off_decoder
```

Note that the default [configurations](./deep_semantic_segmentation/parameter_manager/deeplab/pascal.py) are based on original 
setting, but it requires fairly huge computational resources, so the above option is one way
to compromise the quality and make sure the model can be trained even with small GPU server.
   
 
 


