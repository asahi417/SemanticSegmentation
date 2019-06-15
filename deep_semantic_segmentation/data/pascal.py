import os
import urllib.request
import tarfile
import random
import numpy as np
from glob import glob
from PIL import Image
from ..util import WORK_DIR, create_log


DATA = 'data/pascal'
NUM_CLASS = 21
IGNORE_VALUE = 0

BASE_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
FILENAME = "VOCtrainval_11-May-2012.tar"
GRAY_SEG_OUTPUT_DIR = 'SegmentationClassGray'
LOGGER = create_log()


def download(raw_data_dir: str=None):
    """ download and unzip  """

    if raw_data_dir is None:
        raw_data_dir = os.path.join(WORK_DIR, DATA)

    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir, exist_ok=True)
    url = '%s/%s' % (BASE_URL, FILENAME)
    path_zip = os.path.join(raw_data_dir, FILENAME)
    if not os.path.exists(path_zip):
        LOGGER.info('downloading data: %s -> %s' % (url, path_zip))
        urllib.request.urlretrieve(url, path_zip)

    path_unzip_voc = os.path.join(raw_data_dir, 'VOCdevkit', 'VOC2012')
    if not os.path.exists(path_unzip_voc):
        LOGGER.info('unzipping: %s -> %s' % (path_zip, path_unzip_voc))
        with tarfile.open(path_zip, "r") as zip_ref:
            zip_ref.extractall(raw_data_dir)

    gray_seg_dir = os.path.join(path_unzip_voc, GRAY_SEG_OUTPUT_DIR)
    if not os.path.exists(gray_seg_dir):
        LOGGER.info('convert color to gray scale')
        os.makedirs(gray_seg_dir)
        seg_dir = os.path.join(path_unzip_voc, 'SegmentationClass')
        seg_list = glob(os.path.join(seg_dir, '*.png'))
        convert_rgb_to_gray(seg_list, gray_seg_dir)

    return path_unzip_voc


def convert_rgb_to_gray(file_list, output_dir):
    """ segmentation map has to be gray scale, but VOC Pascal keeps each segmentation as RGB color image
    so convert them into gray scale"""
    for i in file_list:
        file_name = i.split('/')[-1]
        image_array = np.array(Image.open(i))
        image_array = np.rint(image_array).astype('uint8')
        image_pil = Image.fromarray(image_array, 'L')
        image_pil.save(os.path.join(output_dir, file_name))


class BatchFeeder:

    def __init__(self,
                 data_type: str='training',
                 raw_data_dir=None):
        """ Data Batcher

         Usage
        -------------
        >>> batcher = BatchFeeder('training')
        >>> iter_batcher = iter(batcher)
        >>> next(iter_batcher)

         Parameter
        ------------------
        data_type: str
            'training' or 'validation'

        """

        map_to_file_list = dict(training='train.txt', validation='val.txt', )
        path_to_data_dir = download(raw_data_dir)

        def get_files(__type):
            assert __type in map_to_file_list.keys()

            with open(os.path.join(path_to_data_dir, 'ImageSets', 'Segmentation', map_to_file_list[__type]), 'r') as f:
                file_names = f.read().split('\n')[:-1]

            def __check(path):
                if not os.path.exists(path):
                    raise ValueError('%s is not existed' % path)
                else:
                    return path

            random.shuffle(file_names)
            images = [__check(os.path.join(path_to_data_dir, 'JPEGImages', '%s.jpg' % f)) for f in file_names]
            segs = [__check(os.path.join(path_to_data_dir, GRAY_SEG_OUTPUT_DIR, '%s.png' % f)) for f in file_names]
            return images, segs

        self.img, self.ant = get_files(data_type)

    @property
    def data_size(self):
        return len(self.img)

    def __iter__(self):
        self.__data_index = -1
        return self

    def __next__(self):
        """ return path to image and annotation """
        # """ return image (uint8 ndarray), annotation (uint8 ndarray), and shape (width, height) """
        self.__data_index += 1
        if self.__data_index >= len(self.img):
            raise StopIteration
        img_file = self.img[self.__data_index]
        ant_file = self.ant[self.__data_index]
        return img_file, ant_file



