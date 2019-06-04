""" DeepLab v3+ tensorflow implementation"""

import os
import tensorflow as tf
import numpy as np
from .finetune_models.feature_extractor import DeepImageFeature
from . import util_tf
from ..data import TFRecord
from ..util import create_log, load_finetune_model
from ..common_options import Options

slim = tf.contrib.slim


class DeepLab3:

    def __init__(self,
                 checkpoint: str,
                 **kwargs):
        self.__logger = create_log()
        self.__logger.info(__doc__)

        self.__option = Options(**kwargs)

        self.__checkpoint = checkpoint
        self.__checkpoint_finetune = load_finetune_model(self.__option.model_variant)
        self.__iterator = TFRecord(data_name=self.__option.data_name, batch_size=self.__option.batch_size)
        self.__feature = DeepImageFeature(
            model_variant=self.__option.model_variant,
            output_stride=self.__option.output_stride,
            multi_grid=self.__option.multi_grid,
            use_bounded_activation=self.__option.use_bounded_activation
        )

        self.__logger.info('Build Graph: DeepLab v3')
        self.__build_graph()

        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        if os.path.exists('%s.index' % self.__checkpoint):
            # load model
            self.__logger.info('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.__session, self.__checkpoint)
        else:
            self.__session.run(tf.global_variables_initializer())
            if os.path.exists('%s.index' % self.__checkpoint_finetune):
                # load pre-trained model only
                self.__logger.info('load variable from %s' % self.__checkpoint_finetune)
                self.__saver_backbone.restore(self.__session, self.__checkpoint_finetune)
            else:
                self.__logger.info('start from scratch')

    def __if_batch_norm(self, is_training_tensor):
        batch_norm_tensor = tf.convert_to_tensor(self.__option.batch_norm, dtype=tf.bool)
        return tf.math.logical_and(is_training_tensor, batch_norm_tensor)

    def __build_graph(self):
        #########
        # graph #
        #########
        self.__learning_rate = tf.placeholder_with_default(0.0, [], name='learning_rate')
        self.__is_training = tf.placeholder_with_default(True, [], name='is_training')
        is_batch_norm = self.__if_batch_norm(self.__is_training)

        # setup TFRecord iterator and get image/segmentation map
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

        # feature from pre-trained backbone network (ResNet/Xception):
        # (batch, crop_size/output_stride, crop_size/output_stride, 2048)
        feature, variable_endpoints = self.__feature.feature(self.__image, is_training_for_batch_norm=is_batch_norm)
        self.__logger.info(' * feature shape: %s' % feature.shape)

        # aspp feature: (batch, crop_size/output_stride, crop_size/output_stride, 2048)
        aspp_feature = self.__aspp(feature,
                                   is_training=self.__is_training,
                                   is_batch_norm=is_batch_norm)
        self.__logger.info(' * aspp feature shape: %s' % aspp_feature.shape)

        # decoder
        if self.__option.decoder_output_stride is not None:
            final_logit = self.__decoder(aspp_feature,
                                         is_batch_norm=is_batch_norm,
                                         variable_endpoints=variable_endpoints)
            self.__logger.info(' * decoder output shape: %s' % final_logit.shape)
            # output_stride = self.__option.decoder_output_stride
        else:
            final_logit = aspp_feature
            # output_stride = self.__option.output_stride

        # multi-scale class-wise logit
        logit = self.__class_logit(final_logit)
        self.__logger.info(' * logit shape: %s' % logit.shape)
        # scaled_height = util_tf.scale_dimension(self.__option.crop_size_height, 1.0 / output_stride)
        # scaled_width = util_tf.scale_dimension(self.__option.crop_size_width, 1.0 / output_stride)
        # logit = util_tf.resize_bilinear(logit, [scaled_height, scaled_width])
        # self.__logger.info(' * reshaped logit shape: %s' % logit.shape)

        # up-sample logit to be same as segmentation map
        logit = util_tf.resize_bilinear(logit, [self.__iterator.crop_height, self.__iterator.crop_width])
        self.__logger.info(' * reshaped logit shape: %s' % logit.shape)

        self.__prob = tf.nn.softmax(logit)
        self.__logger.info(' * prob shape: %s' % self.__prob.shape)

        self.__prediction = tf.expand_dims(tf.argmax(self.__prob, axis=-1), axis=-1)
        self.__logger.info(' * prediction shape: %s' % self.__prediction.shape)

        ############
        # optimize #
        ############
        self.__loss = self.__pixel_wise_softmax(logit, self.__segmentation)
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # L2 weight decay
        if self.__option.weight_decay != 0.0:
            l2_loss = self.__option.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])
        else:
            l2_loss = 0.0
        self.__loss += l2_loss

        # optimizer
        # global_step = tf.train.get_or_create_global_step()
        optimizer = util_tf.get_optimizer(self.__option.optimizer, self.__learning_rate)
        if self.__option.gradient_clip is None:
            self.__train_op = optimizer.minimize(self.__loss)
        else:
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.__loss, trainable_variables), self.__option.gradient_clip)
            # self.__train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step)
            self.__train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        ###########
        # logging #
        ###########
        with tf.variable_scope('summary'):
            # logging variables
            self.__logger.info('variables')
            n_var = 0
            for var in trainable_variables:
                sh = var.get_shape().as_list()
                self.__logger.info('%s: %s' % (var.name, str(sh)))
                n_var += np.prod(sh)
            self.__logger.info('total variables: %i' % n_var)

            # saver
            self.__saver = tf.train.Saver()
            self.__saver_backbone = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__option.model_variant)
            )

            # scalar summary
            def get_summary(__name):
                return tf.summary.merge([
                    tf.summary.scalar('%s_loss' % __name, self.__loss),
                    # tf.summary.scalar('%s_accuracy' % __name, self.__accuracy),
                ])
            self.__summary_train = get_summary('train')
            self.__summary_valid = get_summary('valid')

            # image summary
            logging_image_number = 5
            self.__summary_images = tf.summary.merge([
                tf.summary.image('image', tf.cast(self.__image, tf.uint8), logging_image_number),
                tf.summary.image('segmentation_predict',
                                 tf.cast(tf.floor(self.__prediction * 255 / self.__iterator.num_class), tf.uint8),
                                 logging_image_number),
                tf.summary.image('segmentation_truth',
                                 tf.cast(tf.floor(self.__segmentation * 255 / self.__iterator.num_class), tf.uint8),
                                 logging_image_number),
            ])

    def __pixel_wise_softmax(self,
                             logit,
                             segmentation):
        assert logit.get_shape().as_list()[1:2] == segmentation.get_shape().as_list()[1:2]
        segmentation = tf.cast(segmentation, tf.int32)
        segmentation = tf.stop_gradient(segmentation)

        # self.__iterator.segmentation_ignore_value

        logit_flatten = tf.reshape(logit, shape=[-1, self.__iterator.num_class])

        segmentation_flatten = tf.reshape(segmentation, shape=[-1])
        segmentation_flatten_one_hot = tf.one_hot(
            segmentation_flatten, self.__iterator.num_class, on_value=1.0, off_value=0.0)

        not_ignore_mask = tf.cast(
            tf.not_equal(segmentation_flatten, self.__iterator.segmentation_ignore_value),
            tf.float32
        )
        # pixel-wise cross entropy
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=segmentation_flatten_one_hot,
            logits=logit_flatten)
        loss = tf.multiply(loss, not_ignore_mask)
        if self.__option.top_k_percent_pixels == 1.0:
            return tf.reduce_sum(loss)
        else:
            with tf.name_scope('pixel_wise_softmax_loss_hard_example_mining'):
                num_pixels = tf.cast(tf.shape(logit)[0], tf.float32)
                # Compute the top_k_percent pixels based on current training step.
                if self.__option.hard_example_mining_step == 0:
                    # Directly focus on the top_k pixels.
                    top_k_pixels = tf.cast(self.__option.top_k_percent_pixels * num_pixels, tf.int32)
                else:
                    # Gradually reduce the mining percent to top_k_percent_pixels.
                    global_step = tf.to_float(tf.train.get_or_create_global_step())
                    ratio = tf.minimum(1.0, global_step / self.__option.hard_example_mining_step)
                    top_k_pixels = tf.cast(
                        (ratio * self.__option.top_k_percent_pixels + (1.0 - ratio)) * num_pixels,
                        tf.int32
                    )
                top_k_loss, _ = tf.nn.top_k(loss,
                                            k=top_k_pixels,
                                            sorted=True,
                                            name='top_k_percent_pixels')
                final_loss = tf.reduce_sum(top_k_loss)
                num_present = tf.reduce_sum(tf.cast(tf.not_equal(final_loss, 0.0)), tf.float32)
                final_loss = final_loss/(num_present + 1e-8)
                return final_loss

    def __class_logit(self,
                      feature,
                      reuse: bool=False):
        """ Logit for prediction.
        The underlying model is branched out in the last layer when atrous
        spatial pyramid pooling is employed, and all branches are sum-merged
        to form the final logits.
        """

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(self.__option.weight_decay),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                reuse=reuse):
            with tf.variable_scope('logits'):
                branch_logits = []
                for i, rate in enumerate(self.__option.atrous_rates):
                    branch_logits.append(
                        slim.conv2d(
                            feature,
                            num_outputs=self.__iterator.num_class,
                            kernel_size=1,
                            rate=rate,
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='conv_%i' % i))

                logit_sum = tf.add_n(branch_logits)
        return logit_sum

    def __decoder(self,
                  feature,
                  is_batch_norm,
                  variable_endpoints,
                  reuse: bool=False):
        """ Decoder with low-level-feature as skip connection

         Parameter
        ----------------
        feature:
        is_batch_norm:
        variable_endpoints:
        reuse: bool

         Return
        ----------------
        """
        # batch norm config
        batch_norm_params = {
            'is_training': is_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }

        activation_fn = tf.nn.relu6 if self.__option.use_bounded_activation else tf.nn.relu
        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                weights_regularizer=slim.l2_regularizer(self.__option.weight_decay),
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm,
                padding='SAME',
                stride=1,
                reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.variable_scope('decoder'):
                    # Low level feature to be added to the decoder as skip connection.
                    # Of `low_level_feature_endpoints` is not a single layer, it will be U-Net/SegNet like architecture.
                    low_level_feature_endpoints = self.__feature.decoder_end_points[self.__option.decoder_output_stride]
                    for n, __end_point in enumerate(low_level_feature_endpoints):
                        with tf.variable_scope('layer_%i' % n):
                            # Projection low_level_feature.
                            low_level_feature = variable_endpoints[__end_point]
                            low_level_feature = slim.conv2d(low_level_feature,
                                                            num_outputs=48,
                                                            kernel_size=1,
                                                            scope='feature_projection')
                            # At this point, feature maps (feature/low_level_feature) have to be
                            # original_size/self.__option.decoder_output_stride, so reshape to be that shape.
                            scaled_height = util_tf.scale_dimension(self.__option.crop_size_height,
                                                                    1.0/self.__option.decoder_output_stride)
                            scaled_width = util_tf.scale_dimension(self.__option.crop_size_width,
                                                                   1.0 / self.__option.decoder_output_stride)
                            low_level_feature = util_tf.resize_bilinear(low_level_feature, [scaled_height, scaled_width])
                            feature = util_tf.resize_bilinear(feature, [scaled_height, scaled_width])

                            # concat and projection
                            feature = tf.concat([feature, low_level_feature], 3)
                            num_convs = 2
                            for i in range(num_convs):
                                if self.__option.decoder_with_separable_conv:
                                    feature = util_tf.split_separable_conv2d(
                                        feature,
                                        filters=self.__option.depth,
                                        rate=1,
                                        # weight_decay=self.__option.weight_decay,
                                        scope='decoder_conv_%i' % i)
                                else:
                                    feature = slim.conv2d(
                                        feature,
                                        num_outputs=self.__option.depth,
                                        kernel_size=3,
                                        scope='decoder_conv_%i' % i)
        return feature

    def __aspp(self,
               feature,
               is_training,
               is_batch_norm,
               reuse: bool=False):
        """ ASPP (Atrous Spatial Pyramid Pooling) layer + Image Pooling

         Parameter
        --------------
        feature: tensor
            input feature tensor
        is_training: bool tensor
        is_batch_norm: bool tensor
        reuse: bool

         Return
        --------------
        tensor with same height, width and `depth` as channel
        """

        # batch norm config
        batch_norm_params = {
            'is_training': is_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }

        activation_fn = tf.nn.relu6 if self.__option.use_bounded_activation else tf.nn.relu
        height = tf.shape(feature)[1]
        width = tf.shape(feature)[2]
        logits = []

        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                weights_regularizer=slim.l2_regularizer(self.__option.weight_decay),
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm,
                padding='SAME',
                stride=1,
                reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                # image pooling (global average pooling + 1x1 convolution)
                with tf.variable_scope('global_average_pooling'):
                    global_average_pooling = tf.reduce_mean(feature, axis=[1, 2], keepdims=True)
                    image_pooling = slim.conv2d(global_average_pooling, num_outputs=self.__option.depth, kernel_size=1)
                    image_pooling_resize = util_tf.resize_bilinear(
                        image_pooling,
                        [height, width],
                        feature.dtype)
                logits.append(image_pooling_resize)

                # ASPP feature: 1x1 convolution for point-wise feature
                logits.append(slim.conv2d(feature, num_outputs=self.__option.depth, kernel_size=1, scope='aspp_0'))

                if self.__option.atrous_rates:
                    # ASPP feature: 3x3 convolution with different atrous rates
                    for i, rate in enumerate(self.__option.atrous_rates, 1):
                        if self.__option.aspp_with_separable_conv:
                            aspp_features = util_tf.split_separable_conv2d(
                                feature,
                                filters=self.__option.depth,
                                rate=rate,  # rate of dilation/atrous conv
                                # weight_decay=self.__option.weight_decay,
                                scope='aspp_%i' % i)
                        else:
                            aspp_features = slim.conv2d(
                                feature,
                                num_outputs=self.__option.depth,
                                kernel_size=3,
                                rate=rate,
                                scope='aspp_%i' % i)
                        logits.append(aspp_features)

                # Merge branch logits
                concat_logits = tf.concat(logits, axis=3)
                concat_logits = slim.conv2d(
                    concat_logits, num_outputs=self.__option.depth, kernel_size=1, scope='concat_projection')
                concat_logits = slim.dropout(
                    concat_logits,
                    keep_prob=0.9,
                    # rate=0.1,
                    is_training=is_training,
                    scope='dropout')
        return concat_logits

    def train(self,
              epoch):
        pass


    def test(self, is_training=True):
        self.__session.run(self.__initializer, feed_dict={self.__is_training: is_training})
        feature = self.__session.run(self.__prediction, feed_dict={self.__is_training: is_training})
        return feature
