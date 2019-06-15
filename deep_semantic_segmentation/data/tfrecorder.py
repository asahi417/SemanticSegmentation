import os
import tensorflow as tf
from . import ade20k, pascal
from . import image_process_util
from ..util import create_log, TFRECORDS

VALID_DATA_NAME = dict(
    ade20k=dict(
        iterator=ade20k.BatchFeeder,
        ignore_value=ade20k.IGNORE_VALUE,
        num_class=ade20k.NUM_CLASS
    ),
    pascal=dict(
        iterator=pascal.BatchFeeder,
        ignore_value=pascal.IGNORE_VALUE,
        num_class=pascal.NUM_CLASS
    )
)
MEAN_RGB = [123.15, 115.90, 103.06]


def decode_image(content, channels):
    return tf.cond(
        tf.image.is_jpeg(content),
        lambda: tf.cast(tf.image.decode_jpeg(content, channels), tf.float32),
        lambda: tf.cast(tf.image.decode_png(content, channels), tf.float32))


class ShapeCheck:
    """ check shape of (image, segmentation) """

    def __init__(self):

        with tf.Graph().as_default():
            self.__encoded_image = tf.placeholder(dtype=tf.string)
            self.__encoded_segmentation = tf.placeholder(dtype=tf.string)
            self.__session = tf.Session()
            self.__image = decode_image(self.__encoded_image, 3)
            self.__segmentation = decode_image(self.__encoded_segmentation, 1)

    def validate_shape(self, encoded_image, encoded_segmentation):
        """ check shape consistency (height, width) of image and segmentation map """
        image, segmentation = self.__session.run(
            [self.__image, self.__segmentation],
            feed_dict={self.__encoded_image: encoded_image, self.__encoded_segmentation: encoded_segmentation})
        if len(image.shape) != 3 or image.shape[-1] != 3:
            raise ValueError('The image not supported.')
        if len(segmentation.shape) != 3 or segmentation.shape[-1] != 1:
            raise ValueError('The segmentation not supported.')

        h_image, w_image = image.shape[:2]
        h_seg, w_seg = segmentation.shape[:2]
        if h_image != h_seg or w_image != w_seg:
            raise RuntimeError('Shape mismatched between image and label.')


class TFRecord:

    flag = dict(
        image='image/encoded',
        filename='image/filename',
        segmentation='image/segmentation/encoded'
    )

    def __init__(self,
                 crop_height: int,
                 crop_width: int,
                 data_name: str,
                 format_image: str='jpg',
                 format_segmentation: str='png',
                 min_scale_factor: float=0.5,
                 max_scale_factor: float=2.0,
                 scale_factor_step_size: float=0.25,
                 min_resize_value: int=None,
                 max_resize_value: int=None,
                 resize_factor: int=None,
                 batch_size: int=8):
        """ Constructor of TFRecorder """
        if data_name not in VALID_DATA_NAME.keys():
            raise ValueError('undefined data: %s not in %s' % (data_name, list(VALID_DATA_NAME.keys())))
        self.__format_image = format_image
        self.__format_segmentation = format_segmentation
        self.__batch_feeders = VALID_DATA_NAME[data_name]['iterator']
        self.__logger = create_log()
        self.__tfrecord_path = os.path.join(TFRECORDS, data_name)
        if not os.path.exists(self.__tfrecord_path):
            os.makedirs(self.__tfrecord_path, exist_ok=True)

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.scale_factor_step_size = scale_factor_step_size
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.batch_size = batch_size

        # mask white as padding region
        self.segmentation_ignore_value = VALID_DATA_NAME[data_name]['ignore_value']
        self.num_class = VALID_DATA_NAME[data_name]['num_class']

        self.__shape_checker = ShapeCheck()

    @property
    def tfrecord_dir(self):
        return self.__tfrecord_path

    def get_iterator(self, is_training):
        """ get `tf.data.Iterator` and iterator initializer """
        tfrecord_path = tf.cond(
            is_training,
            lambda: os.path.join(self.__tfrecord_path, 'training.tfrecord'),
            lambda: os.path.join(self.__tfrecord_path, 'validation.tfrecord'))
        preprocessing_function = self.image_preprocessing(is_training)
        data_set_api = tf.data.TFRecordDataset(tfrecord_path)
        data_set_api = data_set_api.map(self.parse_tfrecord).map(preprocessing_function)
        data_set_api = data_set_api.shuffle(buffer_size=100)
        data_set_api = data_set_api.repeat(1)
        data_set_api = data_set_api.batch(self.batch_size).prefetch(self.batch_size)
        iterator = data_set_api.make_one_shot_iterator()
        initializer = iterator.make_initializer(data_set_api)
        return iterator, initializer

    def parse_tfrecord(self, example_proto):
        """ Parse tfrecord-encoded data

        image: tf.uint8 -> tf.float32
        segmentation: tf.uint8 -> tf.float32
        filename: tf.byte -> tf.string
        """
        features = {
            self.flag['image']: tf.FixedLenFeature((), tf.string, default_value=''),
            self.flag['filename']: tf.FixedLenFeature((), tf.string, default_value=''),
            self.flag['segmentation']: tf.FixedLenFeature((), tf.string, default_value='')
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        parsed_features[self.flag['image']] = decode_image(parsed_features[self.flag['image']], channels=3)
        parsed_features[self.flag['segmentation']] = decode_image(parsed_features[self.flag['segmentation']], channels=1)
        parsed_features[self.flag['filename']] = tf.cast(parsed_features[self.flag['filename']], tf.string)
        return parsed_features

    def image_preprocessing(self, is_training):
        """ Return image-preprocessing function (if `is_training`, skip some augmentations)
        This will be applied for training dataset to perform data augmentation
        """

        def __image_preprocessing(parsed_tensor):
            image = parsed_tensor[self.flag['image']]
            segmentation = parsed_tensor[self.flag['segmentation']]


            def __resize(__image, __label):
                if self.min_resize_value is not None or self.max_resize_value is not None:
                    __image, __label = image_process_util.resize_to_range(
                            image=__image,
                            label=__label,
                            min_size=self.min_resize_value,
                            max_size=self.max_resize_value,
                            factor=self.resize_factor,
                            align_corners=True)
                return __image, __label

            def __augment_pad(__image, __label):
                """ augment image by mean padding to get [self.crop_height, self.crop_width, channel] """
                image_shape = tf.shape(__image)
                image_height = image_shape[0]
                image_width = image_shape[1]
                target_height = image_height + tf.maximum(self.crop_height - image_height, 0)
                target_width = image_width + tf.maximum(self.crop_width - image_width, 0)
                __image = image_process_util.pad_to_bounding_box(
                    image=__image,
                    offset_height=0,
                    offset_width=0,
                    target_height=target_height,
                    target_width=target_width,
                    pad_value=MEAN_RGB)
                __label = image_process_util.pad_to_bounding_box(
                    image=__label,
                    offset_height=0,
                    offset_width=0,
                    target_height=target_height,
                    target_width=target_width,
                    pad_value=self.segmentation_ignore_value)
                return __image, __label

            def __augment_scale(__image, __label):
                """ augment image by random scaling (only training) """
                scale = image_process_util.get_random_scale(
                    min_scale_factor=self.min_scale_factor,
                    max_scale_factor=self.max_scale_factor,
                    step_size=self.scale_factor_step_size)
                __image_aug, __label_aug = image_process_util.randomly_scale_image_and_label(
                    image=__image,
                    label=__label,
                    scale=scale)

                __image_return = tf.cond(
                    is_training,
                    lambda: __image_aug,
                    lambda: __image)

                __label_return = tf.cond(
                    is_training,
                    lambda: __label_aug,
                    lambda: __label)

                # __image_return = tf.where(is_training, __image_aug, __image)
                # __label_return = tf.where(is_training, __label_aug, __label)
                return __image_return, __label_return

            def __augment_crop(__image, __label):
                """ apply same cropping mask to get image with [self.crop_height, self.crop_width] """
                __image_aug, __label_aug = image_process_util.random_crop(
                    image_list=[__image, __label],
                    crop_height=self.crop_height,
                    crop_width=self.crop_width)
                return __image_aug, __label_aug

            def __augment_flip(__image, __label):
                __image_aug, __label_aug, _ = image_process_util.flip_dim(
                    [__image, __label], 0.5, dim=1)
                __image_return = tf.where(is_training, __image_aug, __image)
                __label_return = tf.where(is_training, __label_aug, __label)
                return __image_return, __label_return

            # preprocess
            image, segmentation = __resize(image, segmentation)
            image, segmentation = __augment_scale(image, segmentation)
            image, segmentation = __augment_pad(image, segmentation)
            image, segmentation = __augment_crop(image, segmentation)
            image, segmentation = __augment_flip(image, segmentation)

            # check size
            image.set_shape([self.crop_height, self.crop_width, 3])
            segmentation.set_shape([self.crop_height, self.crop_width, 1])

            # update feature
            parsed_tensor[self.flag['image']] = image
            parsed_tensor[self.flag['segmentation']] = segmentation
            return parsed_tensor

        return __image_preprocessing

    def convert_to_tfrecord(self, progress_interval: int=1000):
        """ Converting data to TFRecord """

        data_types = self.__batch_feeders.keys()
        self.__logger.info('start converting to tfrecord format')

        for __type in data_types:
            tfrecord_path = os.path.join(self.__tfrecord_path, __type + '.tfrecord')

            if os.path.exists(tfrecord_path):
                inp = input('(tfrecord exists at %s. type `y` to overwrite) >>>' % tfrecord_path)
                if inp != 'y':
                    continue

            self.__logger.info('* data type: %s -> %s' % (__type, tfrecord_path))
            iterator = self.__batch_feeders(__type)
            # iterator = self.__batch_feeders[__type]

            with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
                for n, data in enumerate(iterator):
                    file_image, file_segmentation = data

                    # encode image/segmentation from file path
                    encoded_image_data = tf.gfile.GFile(file_image, 'rb').read()
                    encoded_segmentation_data = tf.gfile.GFile(file_segmentation, 'rb').read()

                    # img_data, ant_data, (w, h) = data
                    self.__shape_checker.validate_shape(encoded_image_data, encoded_segmentation_data)

                    # Convert to tf example.
                    example = self.data_to_tfexample(
                        encoded_image_data=encoded_image_data,
                        encoded_segmentation_data=encoded_segmentation_data,
                        filename_image=file_image,
                    )
                    tfrecord_writer.write(example.SerializeToString())

                    if n % progress_interval == 0:
                        progress_perc = n / iterator.data_size * 100
                        self.__logger.info(' - %i / %i (%0.2f %%)' % (n, iterator.data_size, progress_perc))

    def data_to_tfexample(self,
                          encoded_image_data,
                          encoded_segmentation_data,
                          filename_image
                          ):
        """Converts one image/segmentation pair to tf example.

         Parameter
        -----------------
        encoded_image_data: str
            encoded image
        encoded_segmentation_data: str
            encoded segmentation image
        filename_image: str
            image filename.
        height: int
            Image height.
        width: int
            Image width.

         Return
        -----------------
          tf example of one image/segmentation pair.
        """

        # def int64_list_feature(values):
        #     """ Returns a TF-Feature of int64_list. """
        #     if not isinstance(values, collections.Iterable):
        #         values = [values]
        #     return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

        def bytes_list_feature(values):
            """ Returns a TF-Feature of bytes. """
            if isinstance(values, str):
                values = values.encode()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    self.flag['image']: bytes_list_feature(encoded_image_data),
                    self.flag['filename']: bytes_list_feature(filename_image),
                    self.flag['segmentation']: bytes_list_feature(encoded_segmentation_data),
                }))
