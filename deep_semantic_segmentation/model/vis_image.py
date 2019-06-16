""" Base model  """

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from . import util_tf
from ..parameter_manager import ParameterManager
from ..data import TFRecord
from ..util import create_log, TFRECORDS

slim = tf.contrib.slim


class VisImage:

    def __init__(self,
                 data_name: str,
                 model_name: str='DeepLab',
                 **kwargs):
        self.__logger = create_log()
        self.__logger.info(__doc__)

        self.__option = ParameterManager(model_name=model_name,
                                         data_name=data_name,
                                         debug=True,
                                         **kwargs)

        self.__iterator = TFRecord(data_name=self.__option('data_name'),
                                   crop_height=self.__option('crop_height'),
                                   crop_width=self.__option('crop_width'),
                                   batch_size=self.__option('batch_size'),
                                   min_resize_value=self.__option('min_resize_value'),
                                   max_resize_value=self.__option('max_resize_value'),
                                   resize_factor=self.__option('output_stride'))

        self.__logger.info('Build Graph: Visualization Model')
        self.__build_graph()
        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.__session.run(tf.global_variables_initializer())

    def __build_graph(self):
        self.is_training = tf.placeholder_with_default(True, [])
        iterator, self.initializer = self.__iterator.get_iterator(is_training=self.is_training)
        data = iterator.get_next()

        self.image = data[self.__iterator.flag['image']]
        self.segmentation = data[self.__iterator.flag['segmentation']]
        self.segmentation_color = util_tf.coloring_segmentation(
            self.segmentation,
            [self.__iterator.crop_height, self.__iterator.crop_width])
        self.filename = data[self.__iterator.flag['filename']]

    def get_image(self):
        image, seg, filename = self.__session.run([self.image, self.segmentation_color, self.filename])
        filename = [byte.decode() for byte in filename]
        return image, seg, filename

    def setup(self, is_training: bool):
        self.__session.run(self.initializer, feed_dict={self.is_training: is_training})

    def array_to_image(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(TFRECORDS, self.__option('data_name'), 'examples')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        images, segs, filenames = self.get_image()
        for i in range(len(filenames)):
            image, seg, filename = images[i], segs[i], filenames[i]

            print('image shape       :', image.shape)
            print('segmentation shape:', seg.shape, np.unique(seg))
            print('filenames         :', filename)

            base_name = str(filename).split('/')[-1].replace('.jpg', '.png')
            for array, name in zip([image, seg], ['image', 'seg']):
                img = np.rint(array).astype('uint8')
                img = Image.fromarray(img, 'RGB')
                path_to_save = os.path.join(output_dir, '%s_%s' % (name, base_name))
                print('save to %s' % path_to_save)
                img.save(path_to_save)





