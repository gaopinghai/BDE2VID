'''
This code is provided for internal research and development purposes by Huawei solely,
in accordance with the terms and conditions of the research collaboration agreement of May 7, 2020.
Any further use for commercial purposes is subject to a written agreement.
'''
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from .submodules import \
    ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, ResidualBlock, ConvLSTM, \
    ConvGRU, RecurrentResidualLayer


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def identity(x1, x2=None):
    return x1


class BaseUNet(nn.Module):
    def __init__(self, num_bins, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None,
                 use_upsample_conv=True, kernel_size=5):
        super(BaseUNet, self).__init__()

        self.num_bins = num_bins
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        if self.skip_type == 'sum':
            self.apply_skip_connection = skip_sum
        elif self.skip_type == 'concat':
            self.apply_skip_connection = skip_concat
        elif self.skip_type == 'no_skip' or self.skip_type is None:
            self.apply_skip_connection = identity
        else:
            raise KeyError('Could not identify skip_type, please add "skip_type":'
                           ' "sum", "concat" or "no_skip" to config["model"]')
        self.activation = identity if activation is None or activation == 'identity' else getattr(torch, activation)
        self.norm = norm
        self.kernel_size = kernel_size

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_bins > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = identity if activation is None or activation == 'identity' else getattr(torch, activation)

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)


class UNet(BaseUNet):
    def __init__(self, num_bins, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNet, self).__init__(num_bins, num_output_channels, skip_type, activation,
                                   num_encoders, base_num_channels, num_residual_blocks, norm, use_upsample_conv)

        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(ConvLayer(input_size, output_size, kernel_size=5,
                                           stride=2, padding=2, norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x):
        """
        :param x: N x num_bins x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_bins, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNetRecurrent, self).__init__(num_bins, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type=recurrent_block_type,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, prev_states):
        """
        :param x: N x num_bins x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i])
            blocks.append(x)
            states.append(state)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))

        return img, states


class UNetFire(BaseUNet):
    """
    """

    def __init__(self, num_bins, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convgru', base_num_channels=16,
                 num_residual_blocks=2, norm=None, kernel_size=3,
                 recurrent_blocks={'resblock': [0]}):
        super(UNetFire, self).__init__(num_bins=num_bins,
                                       num_output_channels=num_output_channels,
                                       skip_type=skip_type,
                                       base_num_channels=base_num_channels,
                                       num_residual_blocks=num_residual_blocks,
                                       norm=norm,
                                       kernel_size=kernel_size)

        self.recurrent_blocks = recurrent_blocks
        self.num_recurrent_units = 0
        self.head = RecurrentConvLayer(self.num_bins,
                                       self.base_num_channels,
                                       kernel_size=self.kernel_size,
                                       padding=self.kernel_size // 2,
                                       recurrent_block_type=recurrent_block_type,
                                       norm=self.norm)
        self.num_recurrent_units += 1
        self.resblocks = nn.ModuleList()
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i in range(self.num_residual_blocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                self.resblocks.append(RecurrentResidualLayer(
                    in_channels=self.base_num_channels,
                    out_channels=self.base_num_channels,
                    recurrent_block_type=recurrent_block_type,
                    norm=self.norm))
                self.num_recurrent_units += 1
            else:
                self.resblocks.append(ResidualBlock(self.base_num_channels,
                                                    self.base_num_channels,
                                                    norm=self.norm))

        self.pred = ConvLayer(2 * self.base_num_channels if self.skip_type == 'concat' else self.base_num_channels,
                              self.num_output_channels, kernel_size=1, padding=0, activation=None, norm=None)

    def forward(self, x, prev_states):
        """
        :param x: N x num_bins x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        if prev_states is None:
            prev_states = [None] * (self.num_recurrent_units)

        states = []
        state_idx = 0

        # head
        x, state = self.head(x, prev_states[state_idx])
        state_idx += 1
        states.append(state)

        head_feature_map = x

        # residual blocks
        recurrent_indices = self.recurrent_blocks.get('resblock', [])
        for i, resblock in enumerate(self.resblocks):
            if i in recurrent_indices or -1 in recurrent_indices:
                x, state = resblock(x, prev_states[state_idx])
                state_idx += 1
                states.append(state)
            else:
                x = resblock(x)

        # tail
        img = self.pred(x)
        return img, states


class UNetFlow(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        unet_kwargs['num_output_channels'] = 3
        unet_kwargs['activation'] = None
        self.recurrent_block_type = unet_kwargs.pop('recurrent_block_type', None)
        super().__init__(**unet_kwargs)
        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()
        self.states = [None] * self.num_encoders

    def forward(self, x):
        """
        :param x: N x num_bins x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        img_flow = self.pred(self.apply_skip_connection(x, head))

        output_dict = {'image': img_flow[:, 0:1, :, :]}

        return output_dict