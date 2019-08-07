# DeepLab v3+ tensorflow implementation
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)  
Tensorflow implementation of [DeepLab v3+](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf).
Although you can find [the official implementation](https://github.com/tensorflow/models/tree/master/research/deeplab) from Google research,
it took a lot of time to understand the codes mainly because of intensive usage of high level tensorflow API and complex dependency in between each script.
Thus, we decided to re-implement DeepLab v3+ by myself to understand the architecture and see the representation capacity. 
Also we provide minimal configuration by which one can run training on machine with only single GPU, since 
the original configuration in the paper requires pretty high spec machine with multiple GPUs because of the huge model size.
Some parts of this repository, such as scripts of Xception/ResNet and those pre-trained checkpoints, are directory inherited from the official implementation.  

## Get started

```
git clone https://github.com/asahi417/SemanticSegmentation
cd SemanticSegmentation
pip install .
```


## Train model
Train DeeplLab model over some benchmark dataset.
 
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
Once the tfrecord is ready, you train model ([default parameter](./deep_semantic_segmentation/parameter_manager/deeplab)).

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

The checkpoints will be created under `./data/checkpoints/model/DeepLab/` and you can launch tensorboard (`tensorboard --logdir==.`) 
to monitor training progress.

<p align="center">
  <img src="./img/tensorboard_sample_0.png" width="800">
  <br><i>Fig 1: tensorboard view for learning curve </i>
</p>

<p align="center">
  <img src="./img/tensorboard_sample_1.png" width="800">
  <br><i>Fig 2: tensorboard view for sample prediction </i>
</p>

For pascal data, it takes 5 days on single Tesla K80.

## Result
We trained DeepLab with couple of hyperparameter combinations and results are as below.

| model | decoder | backbone     | mIoU | pixel accuracy |
| ----- | ------- | ------------ | ---- | -------------- | 
| **A** | `False` | `Xception41` |    |  | 
| **B** | `False` | `Xception65` |    |  |
| **C** | `True`  | `Xception41` |    |  |
| **D** | `True`  | `Xception65` |    |  |

### Discussion 
<p align="center">
  <img src="./img/123_auexrplozt_krcnnyfrxk.jpg" width="800">
  <br><i>Fig 3: sample prediction </i>
</p>


