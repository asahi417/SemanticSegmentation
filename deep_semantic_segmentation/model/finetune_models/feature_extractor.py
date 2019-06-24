# import functools
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_utils
from . import resnet
from . import xception

slim = tf.contrib.slim


# MODEL_CONFIG
# - arg_scope
#   Model specific arg_scope, which is used as sharing configuration like processing (batch norm, padding, ...)
# - network
#   Network for each model name
# - name
#   A map from feature extractor name to the network name scope used in the
#   ImageNet pretrained versions of these models.
# - preprocess
#   Model specif preprocessing
# - decoder_end_points
#   Scope of low-level feature, which will be used in decoder. A dictionary with keys as decoder stride,
#   but currently only 4 is allowed.
MODEL_CONFIG = {
    'resnet_v1_101':
        {
            'arg_scope': resnet_utils.resnet_arg_scope,
            'network': resnet.resnet_v1_101,
            'name': 'resnet_v1_101',
            'preprocess': resnet.preprocess_subtract_imagenet_mean,
            'decoder_end_points': {4: ['resnet_v1_101/block1/unit_2/bottleneck_v1/conv3']}
        },
    'xception_71':
        {
            'arg_scope': xception.xception_arg_scope,
            'network': xception.xception_71,
            'name': 'xception_71',
            'preprocess': xception.preprocess_zero_mean_unit_range,
            'decoder_end_points': {4: ['xception_71/entry_flow/block3/unit_1/xception_module/separable_conv2_pointwise']}
        },
    'xception_65_coco':
        {
            'arg_scope': xception.xception_arg_scope,
            'network': xception.xception_65,
            'name': 'xception_65',
            'preprocess': xception.preprocess_zero_mean_unit_range,
            'decoder_end_points': {4: ['xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise']}
        },
    'xception_65':
        {
            'arg_scope': xception.xception_arg_scope,
            'network': xception.xception_65,
            'name': 'xception_65',
            'preprocess': xception.preprocess_zero_mean_unit_range,
            'decoder_end_points': {4: ['xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise']}
        },
    'xception_41':
        {
            'arg_scope': xception.xception_arg_scope,
            'network': xception.xception_41,
            'name': 'xception_41',
            'preprocess': xception.preprocess_zero_mean_unit_range,
            'decoder_end_points': {4: ['xception_41/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise']}
        }
}


def get_arg_scope(model_variant, arg_scope_instance, **kwargs):
    if 'resnet' in model_variant:
        return arg_scope_instance(
            batch_norm_decay=0.95,
            batch_norm_epsilon=1e-5,
            batch_norm_scale=True)
    elif 'xception' in model_variant:
        return arg_scope_instance(
            weight_decay=0.0,
            batch_norm_decay=0.9997,
            batch_norm_epsilon=1e-3,
            batch_norm_scale=True,
            use_batch_norm=kwargs['use_batch_norm'],
            regularize_depthwise=False,
            use_bounded_activation=kwargs['use_bounded_activation'])
    else:
        raise ValueError('Unknown model variant %s.' % model_variant)


class DeepImageFeature:

    def __init__(self,
                 model_variant,
                 output_stride: int=8,
                 multi_grid=None,
                 use_bounded_activation=False,
                 finetune_batch_norm: bool=True,
                 weight_deacay: float=0.00004):

        if model_variant not in MODEL_CONFIG.keys():
            raise ValueError('Unknown model variant %s.' % model_variant)

        model_config = MODEL_CONFIG[model_variant]
        self.__output_stride = output_stride
        self.__multi_grid = multi_grid
        self.__finetune_batch_norm = finetune_batch_norm
        self.__num_classes = None  # this results in final logit dimension, and should be None
        self.__arg_scope = get_arg_scope(model_variant,
                                         model_config['arg_scope'],
                                         use_batch_norm=True,
                                         weight_deacay=weight_deacay,
                                         use_bounded_activation=use_bounded_activation)
        self.__network = model_config['network']
        self.__preprocess = model_config['preprocess']
        self.name = model_config['name']
        self.decoder_end_points = model_config['decoder_end_points']

    def feature(self,
                images,
                is_training,
                # is_training_bn,
                reuse=None):
        """
         Parameter
        ----------------
        images: tensor
        is_training_for_batch_norm: bool tensor
        reuse: bool
        """

        # if self.__finetune_batch_norm:
        #     is_training_bn = False

        is_training_bn = tf.logical_and(tf.convert_to_tensor(self.__finetune_batch_norm), is_training)
        with slim.arg_scope(self.__arg_scope):
            feature, endpoint = self.__network(
                self.__preprocess(images, tf.float32),
                num_classes=self.__num_classes,
                is_training_bn=is_training_bn,
                is_training=is_training,
                global_pool=False,
                output_stride=self.__output_stride,
                multi_grid=self.__multi_grid,
                reuse=reuse,
                scope=self.name)

        return feature, endpoint
