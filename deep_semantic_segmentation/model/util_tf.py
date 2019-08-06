import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
# from tensorflow.python.framework import ops

slim = tf.contrib.slim


def get_learning_rate(base_learning_rate: float,
                      decay_method: str = 'poly',
                      learning_power: float =0.9,
                      training_number_of_steps: int = 3000,
                      learning_rate_decay_step: int = 2000,
                      learning_rate_decay_factor: float = 0.1):
    """Gets model's learning rate.

    Computes the model's learning rate for different learning policy.
    Right now, only "step" and "poly" are supported.
    (1) The learning policy for "step" is computed as follows:
      current_learning_rate = base_learning_rate *
        learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
    See tf.train.exponential_decay for details.
    (2) The learning policy for "poly" is computed as follows:
      current_learning_rate = base_learning_rate *
        (1 - global_step / training_number_of_steps) ^ learning_power

     Parameter
    -----------------
    base_learning_rate: The base learning rate for model training.
    decay_method: Learning rate policy for training.
    training_number_of_steps: Number of steps for training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    learning_power: Power used for 'poly' learning policy.

     Returns
    -----------------
    Learning rate for the specified learning policy (tensor).
    """
    # get global step from graph (global_step should be passed to minimizer to be updated every batch)
    global_step = tf.train.get_or_create_global_step()

    if decay_method == 'step':
        return tf.train.exponential_decay(
            base_learning_rate,
            global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
    elif decay_method == 'poly':
        return tf.train.polynomial_decay(
            base_learning_rate,
            global_step,
            training_number_of_steps,
            end_learning_rate=0,
            power=learning_power)
    elif decay_method == 'identity':
        return base_learning_rate
    else:
        raise ValueError('Unknown learning policy.')


def coloring_segmentation(segmentation, shape):
    """ from segmentation map with class label to color images [suppposed to be used in tf graph] """
    [height, width] = shape

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

        tensor_colormap = tf.convert_to_tensor(colormap, tf.int64)
        return tensor_colormap

    color_map = segmentation_colormap()
    vis = tf.cast(segmentation, tf.int64)
    vis_flatten = tf.reshape(vis, shape=[-1])
    vis_flatten_colored = tf.map_fn(lambda x: color_map[x], vis_flatten, dtype=tf.int64)
    vis_flatten_colored = tf.cast(vis_flatten_colored, dtype=tf.uint8)
    batch = dynamic_batch_size(vis)
    vis_colored = tf.reshape(vis_flatten_colored, shape=[batch, height, width, 3])
    return vis_colored


def get_optimizer(optimizer: str, learning_rate: float, **kwargs):
    if optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'adam':
        return tf.train.AdamOptimizer(learning_rate, beta1=0.9, **kwargs)
    elif optimizer == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=0.9, **kwargs)
    else:
        raise ValueError('unknown optimizer: %s' % optimizer)


def get_initializer(initializer: str):
    if initializer == 'variance_scaling':
        return tf.contrib.layers.variance_scaling_initializer()
    elif initializer == 'truncated_normal':
        return tf.initializers.truncated_normal(stddev=0.02)
    elif initializer == 'zero':
        return tf.initializers.zeros()
    else:
        raise ValueError('unknown initializer: %s' % initializer)


def resize_bilinear(images, size, output_dtype=tf.float32):
    """Returns resized images as output_type.

     Parameter
    --------------
    images: A tensor of size [batch, height_in, width_in, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size for the images.
    output_dtype: The destination type.

     Returns
    --------------
      A tensor of size [batch, height_out, width_out, channels] as a dtype of
        output_dtype.
    """
    images = tf.image.resize_bilinear(images, size, align_corners=True)
    return tf.cast(images, dtype=output_dtype)


def scale_dimension(dim, scale):
    """Scales the input dimension.

     Parameter
    --------------
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.

     Returns
    --------------
    Scaled dimension.
    """
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.cast(dim, tf.float32) - 1.0) * scale + 1.0, dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale + 1.0)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
    """ Splits a separable conv2d into depthwise and pointwise conv2d.
    This operation differs from `tf.layers.separable_conv2d` as this operation
    applies activation function between depthwise and pointwise conv2d.

     Parameter
    --------------
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
        of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

     Returns
    --------------
    Computed features after split separable conv2d.
    """

    outputs = slim.separable_conv2d(
        inputs,
        None,
        kernel_size=kernel_size,
        depth_multiplier=1,
        rate=rate,  # rate of dilation/atrous conv
        weights_initializer=tf.truncated_normal_initializer(
            stddev=depthwise_weights_initializer_stddev),
        weights_regularizer=None,
        scope=scope + '_depthwise')
    return slim.conv2d(
        outputs,
        filters,
        kernel_size=1,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=pointwise_weights_initializer_stddev),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        scope=scope + '_pointwise')


def dynamic_batch_size(inputs):
    """ Dynamic batch size, which is able to use in a model without deterministic batch size.
    See https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py
    """
    while nest.is_sequence(inputs):
        inputs = inputs[0]
    return array_ops.shape(inputs)[0]
#
#
# def variable_summaries(var, name):
#     """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
#     https://www.tensorflow.org/guide/summaries_and_tensorboard """
#     with tf.name_scope('var_%s' % name):
#         mean = tf.reduce_mean(var)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#
#         return [tf.summary.scalar('mean', mean),
#                 tf.summary.scalar('stddev', stddev),
#                 tf.summary.scalar('max', tf.reduce_max(var)),
#                 tf.summary.scalar('min', tf.reduce_min(var)),
#                 tf.summary.histogram('histogram', var)]
#
#
# def full_connected(x,
#                    weight_shape,
#                    scope=None,
#                    bias=True,
#                    reuse=None):
#     """ fully connected layer
#     - weight_shape: input size, output size
#     - priority: batch norm (remove bias) > dropout and bias term
#     """
#     with tf.variable_scope(scope or "fully_connected", reuse=reuse):
#         w = tf.get_variable("weight", shape=weight_shape, dtype=tf.float32)
#         x = tf.matmul(x, w)
#         if bias:
#             b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
#             return tf.add(x, b)
#         else:
#             return x
#
#
# def convolution(x,
#                 weight_shape,
#                 stride,
#                 padding="SAME",
#                 scope=None,
#                 bias=True,
#                 reuse=None):
#     """2d convolution
#      Parameter
#     -------------------
#     weight_shape: width, height, input channel, output channel
#     stride (list): [stride for axis 1, stride for axis 2]
#     """
#     with tf.variable_scope(scope or "2d_convolution", reuse=reuse):
#         w = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32)
#         x = tf.nn.conv2d(x, w, strides=[1, stride[0], stride[1], 1], padding=padding)
#         if bias:
#             b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
#             return tf.add(x, b)
#         else:
#             return x