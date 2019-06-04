

class Options:

    crop_size_height = 513
    crop_size_width = 513
    # fine_tune_batch_norm = True
    batch_norm = True

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

    model_variant = 'xception_65'
    batch_size = 4
    data_name = 'ade20k'

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

    def __init__(self, **kwargs):

        if 'model_variant' in kwargs.keys():
            self.model_variant = kwargs['model_variant']
        if 'crop_size_height' in kwargs.keys():
            self.crop_size_height = kwargs['crop_size_height']
        if 'crop_size_width' in kwargs.keys():
            self.crop_size_width = kwargs['crop_size_width']
        if 'batch_norm' in kwargs.keys():
            self.batch_norm = kwargs['batch_norm']
        if 'output_stride' in kwargs.keys():
            self.output_stride = kwargs['output_stride']
        if 'decoder_output_stride' in kwargs.keys():
            self.decoder_output_stride = kwargs['decoder_output_stride']
        if 'multi_grid' in kwargs.keys():
            self.multi_grid = kwargs['multi_grid']
        if 'weight_decay' in kwargs.keys():
            self.weight_decay = kwargs['weight_decay']
        if 'batch_size' in kwargs.keys():
            self.batch_size = kwargs['batch_size']
        if 'data_name' in kwargs.keys():
            self.data_name = kwargs['data_name']
        if 'use_bounded_activation' in kwargs.keys():
            self.use_bounded_activation = kwargs['use_bounded_activation']
        if 'upsample_logits' in kwargs.keys():
            self.upsample_logits = kwargs['upsample_logits']

    def save(self):
        pass
