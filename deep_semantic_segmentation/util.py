import os
import logging
import shutil
import urllib.request
import tarfile
from glob import glob
import numpy as np

# HOME_DIR = os.path.join(os.path.expanduser("~"), 'semantic_segmentation_data')
WORK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data')
TFRECORDS = os.path.join(WORK_DIR, 'tfrecords')
CHECKPOINTS = os.path.join(WORK_DIR, 'checkpoints')
CHECKPOINTS_FINETUNES = dict(
    xception_41=os.path.join(CHECKPOINTS, 'finetune', 'xception_41', 'model.ckpt'),
    xception_65=os.path.join(CHECKPOINTS, 'finetune', 'xception_65', 'model.ckpt'),
    xception_65_coco=os.path.join(CHECKPOINTS, 'finetune', 'xception_65_coco', 'x65-b2u1s2p-d48-2-3x256-sc-cr300k_init.ckpt'),
    xception_71=os.path.join(CHECKPOINTS, 'finetune', 'xception_71', 'model.ckpt')
)
CHECKPOINTS_FINETUNES_URL = dict(
    xception_41='http://download.tensorflow.org/models/xception_41_2018_05_09.tar.gz',
    xception_65='http://download.tensorflow.org/models/deeplabv3_xception_2018_01_04.tar.gz',
    xception_65_coco='http://download.tensorflow.org/models/xception_65_coco_pretrained_2018_10_02.tar.gz',
    xception_71='http://download.tensorflow.org/models/xception_71_2018_05_09.tar.gz'
)


def coloring_segmentation(segmentation):
    """ from segmentation map with class label to color images [non tf] """

    def segmentation_colormap():
        """Creates a segmentation colormap for visualization.

        Returns:
            A Colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3
        return colormap

    colormap_ndarray = segmentation_colormap()
    vis = segmentation.astype(int)
    vis_flatten = np.reshape(vis, [-1]).tolist()
    vis_flatten_colored = np.array([colormap_ndarray[i] for i in vis_flatten], dtype=np.uint8)
    print(vis.shape())
    batch, height, width, _ = list(vis.shape())
    vis_colored = np.reshape(vis_flatten_colored, [batch, height, width, 3])
    return vis_colored


def load_finetune_model(model_variant):

    logger = create_log()
    if model_variant not in CHECKPOINTS_FINETUNES.keys():
        raise ValueError('invalid model: %s not in %s' % (model_variant, list(CHECKPOINTS_FINETUNES.keys())))
    path = CHECKPOINTS_FINETUNES[model_variant]
    dir_ckpt = os.path.dirname(path)
    url = CHECKPOINTS_FINETUNES_URL[model_variant]

    if len(glob(path + '*')) == 0:
        os.makedirs(dir_ckpt, exist_ok=True)
        logger.info('download %s from %s' % (model_variant, url))
        tar_file = os.path.join(dir_ckpt, 'tmp.tar.gz')
        urllib.request.urlretrieve(url, tar_file)

        logger.info('uncompressing file: %s' % tar_file)
        tar = tarfile.open(tar_file, "r:gz")
        tar.extractall(path=dir_ckpt)
        tar.close()

        tar_directory = ''
        for _f in glob(os.path.join(dir_ckpt, '*', '*')):
            tar_directory = os.path.dirname(_f)
            basename = os.path.basename(_f)
            shutil.move(_f, os.path.join(dir_ckpt, basename))

        if len(glob(path + '*')) == 0:
            raise ValueError('no model.ckpt found in given downloaded file')
        os.removedirs(tar_directory)
        os.remove(tar_file)

    return path


def create_log(out_file_path=None, reuse=False):
    """ Logging
        If `out_file_path` is None, only show in terminal
        or else save log file in `out_file_path`
    Usage
    -------------------
    logger.info(message)
    logger.error(error)
    """

    # handler to record log to a log file
    if out_file_path is not None:
        if os.path.exists(out_file_path):
            if not reuse:
                os.remove(out_file_path)
        logger = logging.getLogger(out_file_path)

        if len(logger.handlers) > 1:  # if there are already handler, return it
            return logger
        else:
            # logger.setLevel(logging.DEBUG)
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")

            handler = logging.FileHandler(out_file_path)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            logger_stream = logging.getLogger()
            # check if some stream handlers are already
            if len(logger_stream.handlers) > 0:
                return logger
            else:
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                logger.addHandler(handler)

                return logger
    else:
        # handler to output
        handler = logging.StreamHandler()
        logger = logging.getLogger()

        if len(logger.handlers) > 0:  # if there are already handler, return it
            return logger
        else:  # in case of no, make new output handler
            # logger.setLevel(logging.DEBUG)
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            return logger


if __name__ == '__main__':
    print('WORK_DIR:', WORK_DIR, os.path.exists(WORK_DIR))

