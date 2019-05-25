
import tensorflow as tf
import deep_semantic_segmentation
import argparse
import os

import numpy as np
from PIL import Image


def get_options():
    parser = argparse.ArgumentParser(description='Building TFRecord.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('-d', '--data', help='Dataset.', default='ade20k', type=str, **share_param)
    return parser.parse_args()


class TestTFRecord:

    def __init__(self, data_name):

        self.__iterator = deep_semantic_segmentation.data.TFRecord(data_name=data_name)

        self.__output = os.path.join(self.__iterator.tfrecord_dir, 'examples')
        if not os.path.exists(self.__output):
            os.makedirs(self.__output, exist_ok=True)

        self.__build_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.session.run(tf.global_variables_initializer())

    def __build_graph(self):

        self.is_training = tf.placeholder_with_default(True, [])
        iterator, self.initializer = self.__iterator.get_iterator(is_training=self.is_training)
        data = iterator.get_next()

        self.image = data[self.__iterator.flag['image']]
        self.segmentation = data[self.__iterator.flag['segmentation']]
        self.filename = data[self.__iterator.flag['filename']]

    def get_image(self):
        image, seg, filename = self.session.run([self.image, self.segmentation, self.filename])
        filename = [byte.decode() for byte in filename]
        return image, seg, filename

    def setup(self, is_training: bool):
        self.session.run(self.initializer, feed_dict={self.is_training: is_training})

    def array_to_image(self,
                       array,
                       filename,
                       rgb: bool = False):
        img = np.rint(array).astype('uint8')
        if rgb:
            img = Image.fromarray(img, 'RGB')
        else:
            img = Image.fromarray(img[:, :, 0], 'L')

        path_to_save = os.path.join(self.__output, filename)
        print('save to %s' % path_to_save)
        img.save(path_to_save)


if __name__ == '__main__':
    args = get_options()

    graph = TestTFRecord(args.data)
    while True:
        inp = input('0: Train, 1: Valid >>>')
        if inp == 'q':
            break
        if inp == '0':
            graph.setup(is_training=True)
        elif inp == '1':
            graph.setup(is_training=False)
        else:
            _img, _seg, _fname = graph.get_image()

            print('image shape       :', _img.shape)
            print('segmentation shape:', _seg.shape)
            print('filenames         :', _fname)

            base_name = str(_fname[0]).split('/')[-1].replace('.jpg', '.png')
            graph.array_to_image(_img[0], 'image_' + base_name, rgb=True)
            graph.array_to_image(_seg[0], 'seg_' + base_name)




