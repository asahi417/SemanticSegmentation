""" default parameters """


class Parameter:
    """ Deep Lab v3 (plus) """

    # optimization
    base_learning_rate = 0.00001
    decay_method = 'poly'

    # In order to fine tune the BN layers, one needs to use large batch size (> 12), and set
    # fine_tune_batch_norm = True. If the users have limited GPU memory at hand, please fine-tune
    # from provided checkpoints whose batch norm parameters have been trained, and use smaller
    # learning rate with fine_tune_batch_norm = False.
    batch_norm = False

    # training_number_of_steps = 150000
    # batch_size = 4

    training_number_of_steps = 30000
    batch_size = 4

    # shape
    crop_size_height = 513
    crop_size_width = 513

    # balanced resize in preprocessing
    min_resize_value = None
    max_resize_value = None

    # When using 'xception_65' or 'resnet_v1' model variants, we set
    # atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.output_stride = 16
    output_stride = 16
    atrous_rates = [6, 12, 18]
    # if decoder_output_stride is None, no decoder
    decoder_output_stride = 4

    # ASPP and decoder feature dimension
    depth = 256

    # For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
    # Use 0.0001 for ResNet model variants.
    weight_decay = 0.00004

    model_variant = 'xception_65_coco'
    data_name = 'pascal'

    # Defaults to None. Set multi_grid = [1, 2, 4] when using provided
    # 'resnet_v1_{50,101}_beta' checkpoints.
    multi_grid = None

    use_bounded_activation = False

    aspp_with_separable_conv = True
    decoder_with_separable_conv = True

    # # if True, upsample logit, else downsample segmentation map
    # upsample_logits = True

    # The training step in which exact hard example mining kicks off. Note we
    # gradually reduce the mining percent to the specified
    # top_k_percent_pixels. For example, if hard_example_mining_step=100K and
    # top_k_percent_pixels=0.25, then mining percent will gradually reduce from
    # 100% to 25% until 100K steps after which we only mine top 25% pixels.
    hard_example_mining_step = 0

    # The top k percent pixels (in terms of the loss values) used to compute
    # loss during training. This is useful for hard pixel mining.
    top_k_percent_pixels = 1.0

    optimizer = 'momentum'
    gradient_clip = None
