""" Base model  """

import os
import tensorflow as tf
import numpy as np
from .finetune_models.feature_extractor import DeepImageFeature
from ..data import TFRecord
from ..util import create_log, load_finetune_model
from ..common_options import Options


class BaseModel:

    def __init__(self, **kwargs):
        self.__option = Options(**kwargs)

        self.__checkpoint = load_finetune_model(self.__option.model_variant)
        self.__logger = create_log()
        self.__logger.info('#### Debugging Model ####')
        self.__iterator = TFRecord(data_name=self.__option.data_name, batch_size=self.__option.batch_size)
        self.__feature = DeepImageFeature(
            model_variant=self.__option.model_variant,
            output_stride=self.__option.output_stride,
            multi_grid=self.__option.multi_grid,
            use_bounded_activation=self.__option.use_bounded_activation
        )
        self.__build_graph()

        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        # Load model
        if os.path.exists('%s.index' % self.__checkpoint):
            self.__logger.info('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.__session, self.__checkpoint)
        else:
            self.__logger.info('zero initialization')
            self.__session.run(tf.global_variables_initializer())

    def __build_graph(self):
        # setup TFRecord iterator and get image/segmentation map
        self.__is_training = tf.placeholder_with_default(True, [])
        iterator, self.__initializer = self.__iterator.get_iterator(is_training=self.__is_training)
        data = iterator.get_next()
        image = data[self.__iterator.flag['image']]
        segmentation = data[self.__iterator.flag['segmentation']]

        # input/output placeholder
        self.__image = tf.placeholder_with_default(
            image, [None, self.__iterator.crop_height, self.__iterator.crop_width, 3], name="input_image")
        self.__segmentation = tf.placeholder_with_default(
            segmentation, [None, self.__iterator.crop_height, self.__iterator.crop_width, 1], name="segmentation")

        self.__logger.info(' * image shape: %s' % self.__image.shape)
        self.__feature, self.__endpoint = self.__feature.feature(self.__image,
                                                                 is_training_for_batch_norm=self.__is_training)
        self.__logger.info(' * feature shape: %s' % self.__feature.shape)

        ###########
        # logging #
        ###########
        self.__logger.info('variables')
        n_var = 0
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_variables:
            sh = var.get_shape().as_list()
            self.__logger.info('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)
            # write for tensorboard visualization
            # variable_summaries(var, var.name.split(':')[0].replace('/', '-'))
        self.__logger.info('total variables: %i' % n_var)
        self.__saver = tf.train.Saver()

    def test(self, is_training=True):
        self.__session.run(self.__initializer, feed_dict={self.__is_training: is_training})
        feature = self.__session.run(self.__feature, feed_dict={self.__is_training: is_training})
        return feature



