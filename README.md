# DeepLab v3+ tensorflow implementation
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)  
Tensorflow implementation of [DeepLab v3+](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf).
Although you can find [the official implementation](https://github.com/tensorflow/models/tree/master/research/deeplab) from Google research,
it took me lots of time to understand the codes mainly because of intensive usage of high level tensorflow API and complex dependency in between each script.
Thus, I decided to re-implement DeepLab v3+ by myself to understand the architecture and see the powerful representation capacity. 
Also I provide minimal configuration by which one can run training on machine with only single GPU, since 
the original configuration in the paper requires pretty high spec machine with multiple GPUs because of the huge model size.
Some parts of this repository, such as scripts of Xception and ResNet, are directory inherited from the official implementation.  

## Get started

```
git clone https://github.com/asahi417/SemanticSegmentation
cd SemanticSegmentation
pip install .
```


## Train model
Train DeeplLab model over some benchmark daataset.
 
### Download dataset and converting as TFRecord 
Public benchmark datasets for semantic segmentation, including [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/),
will be downloaded and converted as TFRecord by following script.

```
./bin/build_tfrecord.py [-h] [-d [DATA]]

Encoding to tfrecord.

optional arguments:
  -h, --help            show this help message and exit
  -d [DATA], --data [DATA]
                        Dataset name (`pascal` or `ade20k`)
```      
 
A directory (`./data/data` and `./data/tfrecords`) will be created as the place to store data related files.

### Training
Model can be trained on [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) or [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
by

```
./bin/train.py [-h] [-d [DATA]] [-m [MODEL]] [-b [BATCH_SIZE]]
                [-l [LEARNING_RATE]] [-w [WEIGHT_DECAY]]
                [--aspp_depth [ASPP_DEPTH]] [--crop_size [CROP_SIZE]]
                [--output_stride [OUTPUT_STRIDE]] [--off_decoder]
                [--off_fine_tune_batch_norm] [--backbone [BACKBONE]]
                [--checkpoint [CHECKPOINT]]

optional arguments:
  -h, --help            show this help message and exit
  -d [DATA], --data [DATA]
                        Dataset
  -m [MODEL], --model [MODEL]
                        Model
  -b [BATCH_SIZE], --batch_size [BATCH_SIZE]
                        Batch size
  -l [LEARNING_RATE], --learning_rate [LEARNING_RATE]
                        learning rate
  -w [WEIGHT_DECAY], --weight_decay [WEIGHT_DECAY]
                        weight decay
  --aspp_depth [ASPP_DEPTH]
                        aspp depth
  --crop_size [CROP_SIZE]
                        crop size
  --output_stride [OUTPUT_STRIDE]
                        output_stride
  --off_decoder         unuse decoder
  --off_fine_tune_batch_norm
                        off_fine_tune_batch_norm
  --backbone [BACKBONE]
                        backbone network
  --checkpoint [CHECKPOINT]
                        checkpoint

```

### Result
  

### Train model
After producing TFRecord file, one can start training DeepLav by 

```
python ./bin/train.py -d pascal --crop_size 321 --aspp_depth 128 --backbone xception_41 -b 12 --off_decoder
```

Note that the default [configurations](./deep_semantic_segmentation/parameter_manager/deeplab/pascal.py) are based on original 
setting, but it requires fairly huge computational resources, so the above option is one way
to compromise the quality and make sure the model can be trained even with small GPU server.
   
 
 


