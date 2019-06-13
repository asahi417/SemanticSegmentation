import os
import urllib.request
import tarfile
import random
from glob import glob
# import numpy as np
# from PIL import Image
from ..util import WORK_DIR, create_log


DATA = 'data/pascal'
NUM_CLASS = 21
IGNORE_VALUE = 255

BASE_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012"
FILENAME = "VOCtrainval_11-May-2012.tar"
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

    path_unzip = path_zip.split('.')[0]
    if not os.path.exists(path_unzip):
        LOGGER.info('unzipping: %s -> %s' % (path_zip, path_unzip))
        with tarfile.open(path_zip, "r") as zip_ref:
            zip_ref.extractall(raw_data_dir)
    return path_unzip


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
        assert data_type in ['training', 'validation']

        path_to_data_dir = download(raw_data_dir)

        def get_files(__type):

            def __check(path):
                if not os.path.exists(path):
                    raise ValueError('%s is not existed' % path)
                else:
                    return path

            img_names = glob(os.path.join(path_to_data_dir, 'images', __type, '*.jpg'))
            random.shuffle(img_names)
            ant_names = [
                __check(
                    os.path.join(
                        path_to_data_dir,
                        'annotations',
                        __type,
                        os.path.basename(f).replace('.jpg', '.png')
                    )
                ) for f in img_names]
            return img_names, ant_names

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



