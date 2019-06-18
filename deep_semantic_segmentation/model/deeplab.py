""" DeepLab v3+ tensorflow implementation"""

import os
import tensorflow as tf
import numpy as np
from .finetune_models.feature_extractor import DeepImageFeature
from . import util_tf
from ..parameter_manager import ParameterManager
from ..data import TFRecord
from ..util import create_log, load_finetune_model

slim = tf.contrib.slim

LOG_VAR = os.getenv('LOG_LEVEL', None)
LOG_IMAGE_NUM = os.getenv('LOG_IMAGE_NUM', 3)

class DeepLab:

    def __init__(self,
                 data_name: str,
                 checkpoint: str = None,
                 checkpoint_version: int = None,
                 **kwargs):
        self.__logger = create_log()
        self.__logger.info(__doc__)

        self.__option = ParameterManager(model_name='DeepLab',
                                         data_name=data_name,
                                         checkpoint_dir=checkpoint,
                                         checkpoint_version=checkpoint_version,
                                         **kwargs)

        self.__checkpoint = checkpoint
        self.__checkpoint_finetune = load_finetune_model(self.__option('model_variant'))
        self.__iterator = TFRecord(data_name=self.__option('data_name'),
                                   crop_height=self.__option('crop_height'),
                                   crop_width=self.__option('crop_width'),
                                   batch_size=self.__option('batch_size'),
                                   min_resize_value=self.__option('min_resize_value'),
                                   max_resize_value=self.__option('max_resize_value'),
                                   resize_factor=self.__option('output_stride'))
        self.__feature = DeepImageFeature(
            model_variant=self.__option('model_variant'),
            output_stride=self.__option('output_stride'),
            multi_grid=self.__option('multi_grid'),
            use_bounded_activation=self.__option('use_bounded_activation'),
            finetune_batch_norm=self.__option('fine_tune_batch_norm')
        )

        self.__logger.info('Build Graph: DeepLab')
        self.__logger.info(' - checkpoint: %s' % self.__option.checkpoint_dir)
        self.__build_graph()

        self.__session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.__writer = tf.summary.FileWriter('%s/summary' % self.__option.checkpoint_dir, self.__session.graph)

        if os.path.exists('%s/model.ckpt.index' % self.__option.checkpoint_dir):
            # load model
            self.__logger.info('load variable from %s' % self.__checkpoint)
            self.__saver.restore(self.__session, '%s/model.ckpt' % self.__option.checkpoint_dir)
            self.__warm_start = True
        else:
            self.__session.run(tf.global_variables_initializer())
            if os.path.exists('%s.index' % self.__checkpoint_finetune):
                # load pre-trained model only
                self.__logger.info('load variable from %s' % self.__checkpoint_finetune)
                self.__saver_backbone.restore(self.__session, self.__checkpoint_finetune)
            else:
                raise ValueError('backbone network is not found')
            self.__warm_start = False

    def __build_graph(self):
        #########
        # graph #
        #########
        self.__is_training = tf.placeholder_with_default(True, [], name='is_training')

        # setup TFRecord iterator and get image/segmentation map
        iterator, self.__init_op_iterator = self.__iterator.get_iterator(is_training=self.__is_training)
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
        feature, variable_endpoints = self.__feature.feature(self.__image, is_training=self.__is_training)
        self.__logger.info(' * feature shape: %s' % feature.shape)

        # aspp feature: (batch, crop_size/output_stride, crop_size/output_stride, 2048)
        aspp_feature = self.__aspp(feature, is_training=self.__is_training, )
        self.__logger.info(' * aspp feature shape: %s' % aspp_feature.shape)

        # decoder
        if self.__option('decoder_output_stride') is not None:
            final_logit = self.__decoder(aspp_feature,
                                         is_training=self.__is_training,
                                         variable_endpoints=variable_endpoints)
            self.__logger.info(' * decoder output shape: %s' % final_logit.shape)
        else:
            final_logit = aspp_feature

        # class-wise logit
        logit = self.__class_logit(final_logit)
        self.__logger.info(' * logit shape: %s' % logit.shape)

        # up-sample logit to be same as segmentation map
        logit = util_tf.resize_bilinear(logit, [self.__iterator.crop_height, self.__iterator.crop_width])
        self.__logger.info(' * bilinear-resized logit shape: %s' % logit.shape)
        self.__prob = tf.nn.softmax(logit)
        self.__logger.info(' * prob shape: %s' % self.__prob.shape)
        self.__prediction = tf.cast(tf.expand_dims(tf.argmax(self.__prob, axis=-1), axis=-1), tf.int64)
        self.__logger.info(' * prediction shape: %s' % self.__prediction.shape)

        ############
        # optimize #
        ############
        self.__loss = self.__pixel_wise_softmax(logit, self.__segmentation)
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # L2 weight decay
        if self.__option('weight_decay') != 0.0:
            l2_loss = self.__option('weight_decay') * tf.add_n([tf.nn.l2_loss(v) for v in trainable_variables])
            # l2_loss = 0.0
        else:
            l2_loss = 0.0
        self.__loss += l2_loss

        # global step, which will be increased by one every time the minimizer is called
        # (https://stackoverflow.com/questions/41166681/what-does-global-step-mean-in-tensorflow)
        self.__global_step = tf.train.get_or_create_global_step()
        learning_rate = util_tf.get_learning_rate(
            self.__option('base_learning_rate'),
            decay_method=self.__option('decay_method'),
            training_number_of_steps=self.__option('training_number_of_steps'))

        # optimizer
        optimizer = util_tf.get_optimizer(self.__option('optimizer'), learning_rate)
        if self.__option('gradient_clip') is None:
            self.__train_op = optimizer.minimize(self.__loss, global_step=self.__global_step)
        else:
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.__loss, trainable_variables), self.__option('gradient_clip'))
            self.__train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.__global_step)

        ######################
        # evaluation metrics #
        ######################
        with tf.variable_scope('summary'):
            self.__miou, self.__pixel_accuracy, self.__update_op_metric, self.__init_op_metric \
                = self.__get_metrics(self.__prediction, self.__segmentation)
            # summary to be shown every update/training step
            self.__update_summary = tf.summary.merge([
                tf.summary.scalar('loss', self.__loss),
                tf.summary.scalar('global_step', self.__global_step),
                tf.summary.scalar('learning_rate', learning_rate)
            ])

            def get_summary(__name):
                vis_label = util_tf.coloring_segmentation(self.__segmentation,
                                                          [self.__iterator.crop_height, self.__iterator.crop_width])
                vis_pred = util_tf.coloring_segmentation(self.__prediction,
                                                         [self.__iterator.crop_height, self.__iterator.crop_width])
                __summary_img = tf.summary.merge([
                    tf.summary.image('%s_image' % __name,
                                     tf.cast(self.__image, tf.uint8),
                                     LOG_IMAGE_NUM),
                    tf.summary.image('%s_segmentation_predict' % __name,
                                     vis_pred,
                                     LOG_IMAGE_NUM),
                    tf.summary.image('%s_segmentation_truth' % __name,
                                     vis_label,
                                     LOG_IMAGE_NUM)])
                __summary = tf.summary.merge([
                    tf.summary.scalar('%s_pixel_accuracy' % __name, self.__pixel_accuracy),
                    tf.summary.scalar('%s_miou' % __name, self.__miou),
                    # tf.summary.scalar('%s_loss' % __name, self.__loss),
                ])
                return __summary, __summary_img

            self.__summary_train, self.__summary_img_train = get_summary('train')
            self.__summary_valid, self.__summary_img_valid = get_summary('valid')

            # saver
            self.__saver = tf.train.Saver()
            self.__saver_backbone = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__feature.name)
            )

        ###########
        # logging #
        ###########
        # logging variables
        self.__logger.info('variables')
        n_var = 0
        for var in trainable_variables:
            sh = var.get_shape().as_list()
            if LOG_VAR is None:
                self.__logger.info('%s: %s' % (var.name, str(sh)))
            n_var += np.prod(sh)
        self.__logger.info('total variables: %i' % n_var)

    def __get_metrics(self,
                      prediction,
                      segmentation):
        # tf.metric, which returns `metric` and `update_op`
        # - update_op has to be updated every training step
        # - metric consists of the value averaging over the past results stored by update_op
        scope = 'evaluation_metrics'

        segmentation = tf.cast(segmentation, tf.int64)
        segmentation = tf.stop_gradient(segmentation)
        segmentation_flatten = tf.reshape(segmentation, shape=[-1])
        not_ignore_mask = tf.not_equal(segmentation_flatten, self.__iterator.segmentation_ignore_value)
        segmentation_flatten = segmentation_flatten * tf.cast(not_ignore_mask, tf.int64)

        prediction_flatten = tf.cast(tf.reshape(prediction, shape=[-1]), tf.int64)
        prediction_flatten = prediction_flatten * tf.cast(not_ignore_mask, tf.int64)

        # mean IoU (intersection over union)
        miou, update_op_miou = tf.metrics.mean_iou(
            predictions=prediction_flatten,
            labels=segmentation_flatten,
            weights=tf.cast(not_ignore_mask, tf.float32),
            name=scope,
            num_classes=self.__iterator.num_class)
        # pixel accuracy
        pixel_accuracy, update_op_acc = tf.metrics.accuracy(
            predictions=prediction_flatten,
            labels=segmentation_flatten,
            weights=tf.cast(not_ignore_mask, tf.float32),
            name=scope)
        update_op_metric = tf.group(*[update_op_miou, update_op_acc])
        init_op_metric = tf.variables_initializer(
            var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='summary'),
            name=scope)
        return miou, pixel_accuracy, update_op_metric, init_op_metric

    def __pixel_wise_softmax(self,
                             logit,
                             segmentation):
        assert logit.get_shape().as_list()[1:2] == segmentation.get_shape().as_list()[1:2]
        segmentation = tf.cast(segmentation, tf.int64)
        segmentation = tf.stop_gradient(segmentation)
        segmentation_flatten = tf.reshape(segmentation, shape=[-1])
        not_ignore_mask = tf.not_equal(segmentation_flatten, self.__iterator.segmentation_ignore_value)

        segmentation_flatten = segmentation_flatten * tf.cast(not_ignore_mask, tf.int64)
        segmentation_flatten_one_hot = tf.one_hot(
            segmentation_flatten, self.__iterator.num_class, on_value=1.0, off_value=0.0
        )

        logit_flatten = tf.reshape(logit, shape=[-1, self.__iterator.num_class])
        # logit_flatten = logit_flatten * tf.cast(not_ignore_mask, tf.float32)
        # pixel-wise cross entropy
        if self.__option('top_k_percent_pixels') == 1.0:
            loss = tf.losses.softmax_cross_entropy(
                segmentation_flatten_one_hot,
                logit_flatten,
                weights=tf.cast(not_ignore_mask, tf.float32))
            return loss
        else:
            # batch size
            loss = tf.losses.softmax_cross_entropy(
                segmentation_flatten_one_hot,
                logit_flatten,
                weights=not_ignore_mask,
                reduction=None)
            with tf.name_scope('pixel_wise_softmax_loss_hard_example_mining'):
                num_pixels = tf.cast(tf.shape(logit)[0], tf.float32)
                # Compute the top_k_percent pixels based on current training step.
                if self.__option('hard_example_mining_step') == 0:
                    # Directly focus on the top_k pixels.
                    top_k_pixels = tf.cast(self.__option('top_k_percent_pixels') * num_pixels, tf.int32)
                else:
                    # Gradually reduce the mining percent to top_k_percent_pixels.
                    global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
                    ratio = tf.minimum(1.0, global_step / self.__option('hard_example_mining_step'))
                    top_k_pixels = tf.cast(
                        (ratio * self.__option('top_k_percent_pixels') + (1.0 - ratio)) * num_pixels,
                        tf.int32
                    )
                top_k_loss, _ = tf.nn.top_k(loss,
                                            k=top_k_pixels,
                                            sorted=True,
                                            name='top_k_percent_pixels')
                final_loss = tf.reduce_sum(top_k_loss)
                num_present = tf.reduce_sum(tf.cast(tf.not_equal(final_loss, 0.0)), tf.float32)
                final_loss = final_loss/(num_present + 1e-6)
                return final_loss

    def __class_logit(self,
                      feature,
                      reuse: bool=False):
        """ Final logit with class-lab """

        with slim.arg_scope(
                [slim.conv2d],
                # weights_regularizer=slim.l2_regularizer(self.__option('weight_decay')),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                reuse=reuse):
            with tf.variable_scope('logits'):
                final_logit = slim.conv2d(
                    feature,
                    num_outputs=self.__iterator.num_class,
                    kernel_size=1,
                    rate=1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='conv')
        return final_logit

    def __decoder(self,
                  feature,
                  is_training,
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
        decoded feature map
        """

        scopes = dict(
            # weights_regularizer=slim.l2_regularizer(self.__option('weight_decay')),
            activation_fn=tf.nn.relu6 if self.__option('use_bounded_activation') else tf.nn.relu,
            padding='SAME',
            stride=1,
            reuse=reuse
        )
        # batch norm config
        if self.__option('decoder_batch_norm'):
            batch_norm_values = [slim.batch_norm]
            batch_norm_scopes = dict(
                is_training=is_training,
                decay=0.9997,
                epsilon=1e-5,
                scale=True,
            )
            scopes['normalizer_fn'] = slim.batch_norm
        else:
            batch_norm_values = list()
            batch_norm_scopes = dict()

        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], **scopes):
            with slim.arg_scope(batch_norm_values, **batch_norm_scopes):
                with tf.variable_scope('decoder'):
                    # Low level feature to be added to the decoder as skip connection.
                    # Of `low_level_feature_endpoints` is not a single layer, it will be U-Net/SegNet like architecture.
                    low_level_feature_endpoints = self.__feature.decoder_end_points[self.__option('decoder_output_stride')]
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
                            scaled_height = util_tf.scale_dimension(self.__option('crop_height'),
                                                                    1.0 / self.__option('decoder_output_stride'))
                            scaled_width = util_tf.scale_dimension(self.__option('crop_width'),
                                                                   1.0 / self.__option('decoder_output_stride'))
                            low_level_feature = util_tf.resize_bilinear(low_level_feature, [scaled_height, scaled_width])
                            feature = util_tf.resize_bilinear(feature, [scaled_height, scaled_width])

                            # concat and projection
                            feature = tf.concat([feature, low_level_feature], 3)
                            num_convs = 2
                            for i in range(num_convs):
                                if self.__option('decoder_with_separable_conv'):
                                    feature = util_tf.split_separable_conv2d(
                                        feature,
                                        filters=self.__option('depth'),
                                        rate=1,
                                        # weight_decay=self.__option.weight_decay,
                                        scope='decoder_conv_%i' % i)
                                else:
                                    feature = slim.conv2d(
                                        feature,
                                        num_outputs=self.__option('depth'),
                                        kernel_size=3,
                                        scope='decoder_conv_%i' % i)
        return feature

    def __aspp(self,
               feature,
               is_training,
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

        scopes = dict(
            # weights_regularizer=slim.l2_regularizer(self.__option('weight_decay')),
            activation_fn=tf.nn.relu6 if self.__option('use_bounded_activation') else tf.nn.relu,
            padding='SAME',
            stride=1,
            reuse=reuse
        )
        # batch norm config
        if self.__option('aspp_batch_norm'):
            batch_norm_values = [slim.batch_norm]
            batch_norm_scopes = dict(
                is_training=is_training,
                decay=0.9997,
                epsilon=1e-5,
                scale=True,
            )
            scopes['normalizer_fn'] = slim.batch_norm
        else:
            batch_norm_values = list()
            batch_norm_scopes = dict()

        height = tf.shape(feature)[1]
        width = tf.shape(feature)[2]
        logits = []

        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], **scopes):
            with slim.arg_scope(batch_norm_values, **batch_norm_scopes):
                # image pooling (global average pooling + 1x1 convolution)
                with tf.variable_scope('global_average_pooling'):
                    global_average_pooling = tf.reduce_mean(feature, axis=[1, 2], keepdims=True)
                    image_pooling = slim.conv2d(global_average_pooling, num_outputs=self.__option('depth'), kernel_size=1)
                    self.__logger.info(' * aspp GAP: %s' % image_pooling.shape)
                    image_pooling_resize = util_tf.resize_bilinear(
                        image_pooling,
                        [height, width],
                        feature.dtype)
                    self.__logger.info(' * aspp GAP resized: %s' % image_pooling_resize.shape)
                logits.append(image_pooling_resize)

                # ASPP feature: 1x1 convolution for point-wise feature
                logits.append(slim.conv2d(feature, num_outputs=self.__option('depth'), kernel_size=1, scope='aspp_0'))

                if self.__option('atrous_rates'):
                    # ASPP feature: 3x3 convolution with different atrous rates
                    for i, rate in enumerate(self.__option('atrous_rates'), 1):
                        if self.__option('aspp_with_separable_conv'):
                            aspp_features = util_tf.split_separable_conv2d(
                                feature,
                                filters=self.__option('depth'),
                                rate=rate,  # rate of dilation/atrous conv
                                # weight_decay=self.__option.weight_decay,
                                scope='aspp_%i' % i)
                        else:
                            aspp_features = slim.conv2d(
                                feature,
                                num_outputs=self.__option('depth'),
                                kernel_size=3,
                                rate=rate,
                                scope='aspp_%i' % i)
                        self.__logger.info(' * aspp %i: %s' % (i, aspp_features.shape))
                        logits.append(aspp_features)

                # Merge branch logits
                concat_logits = tf.concat(logits, axis=3)
                self.__logger.info(' * aspp concat: %s' % concat_logits.shape)
                concat_logits = slim.conv2d(
                    concat_logits,
                    num_outputs=self.__option('depth'),
                    kernel_size=1,
                    scope='concat_projection')
                concat_logits = slim.dropout(
                    concat_logits,
                    keep_prob=0.9,
                    # rate=0.1,
                    is_training=is_training,
                    scope='dropout')
        return concat_logits

    def train(self):
        """ Model training method. Logs are all saved in tensorboard.
        - Every epoch, store segmentation result as image (training/validation)
        - Every epoch, store and show metric (training/validation)
        - Every step, store loss, global_step, learning_rate (training)
        """
        logger = create_log(os.path.join(self.__option.checkpoint_dir, 'train.log'))
        step = 0
        loss = 0

        try:
            logger.info('## start training ##')
            while True:
                logger.info('Step: %i' % step)

                ############
                # TRAINING #
                ############
                logger.info(' * training')
                feed_dict = {self.__is_training: True}
                logger.info('  - initialization')
                self.__session.run([self.__init_op_iterator, self.__init_op_metric], feed_dict=feed_dict)

                if step != 0:
                    # this process takes about 5 min
                    logger.info('  - writing images to tensorboard')
                    self.__writer.add_summary(
                        self.__session.run(self.__summary_img_train, feed_dict=feed_dict), global_step=step)

                logger.info('  - training start')
                print()
                while True:
                    try:
                        _, _, summary, step, tmp_loss = self.__session.run(
                            [self.__train_op, self.__update_op_metric, self.__update_summary, self.__global_step, self.__loss],
                            feed_dict=feed_dict)
                        print('   - step: %i, loss: %0.5f (%0.5f) \r' % (step, tmp_loss, tmp_loss-loss), end='', flush=False)
                        loss = tmp_loss
                        self.__writer.add_summary(summary, global_step=step)

                    except tf.errors.OutOfRangeError:
                        print()
                        summary, pix_acc, miou, step = self.__session.run(
                            [self.__summary_train, self.__pixel_accuracy, self.__miou, self.__global_step],
                            feed_dict=feed_dict)
                        self.__writer.add_summary(summary, global_step=step)
                        logger.info('  - [train] pixel accuracy: %0.4f' % pix_acc)
                        logger.info('  - [train] mean IoU      : %0.4f' % miou)
                        break

                #########
                # VALID #
                #########
                logger.info(' * validation')
                feed_dict = {self.__is_training: False}
                logger.info('  - initialization')
                self.__session.run([self.__init_op_iterator, self.__init_op_metric], feed_dict=feed_dict)
                logger.info('  - writing images to tensorboard')
                self.__writer.add_summary(
                    self.__session.run(self.__summary_img_valid, feed_dict=feed_dict), global_step=step)
                logger.info('  - validation start')
                while True:
                    try:
                        self.__session.run(self.__update_op_metric, feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        summary, pix_acc, miou, step = self.__session.run(
                            [self.__summary_valid, self.__pixel_accuracy, self.__miou, self.__global_step],
                            feed_dict=feed_dict)
                        self.__writer.add_summary(summary, global_step=step)
                        logger.info('  - [valid] pixel accuracy: %0.4f' % pix_acc)
                        logger.info('  - [valid] mean IoU      : %0.4f' % miou)
                        break

                # check training status
                step = self.__session.run(self.__global_step)
                if step > self.__option('training_number_of_steps'):
                    logger.info('>>> Training has been completed (current step exceeds training step: %i > %i) . <<<'
                                % (step, self.__option('training_number_of_steps')))
                    break

        except KeyboardInterrupt:
            logger.info('>>> KeyboardInterrupt <<<')

        logger.info('Save checkpoints...')
        self.__saver.save(self.__session, os.path.join(self.__option.checkpoint_dir, 'model.ckpt'))

