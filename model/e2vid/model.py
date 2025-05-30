'''
This code is provided for internal research and development purposes by Huawei solely,
in accordance with the terms and conditions of the research collaboration agreement of May 7, 2020.
Any further use for commercial purposes is subject to a written agreement.
'''
import numpy as np
from .base_model import BaseModel
import torch.nn as nn
import torch
from .unet import UNet, UNetRecurrent, UNetFire, UNetFlow
from os.path import join
from model.submodules import \
    ConvLSTM, ResidualBlock, ConvLayer, ConvGRU, \
    UpsampleConvLayer, TransposedConvLayer
import logging


class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        assert('num_bins' in config)
        self.num_bins = int(config['num_bins'])  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True

        self.kernel_size = int(config.get('kernel_size', 5))


class E2VID(BaseE2VID):
    def __init__(self, config):
        super(E2VID, self).__init__(config)

        self.unet = UNet(num_bins=self.num_bins,
                         num_output_channels=1,
                         skip_type=self.skip_type,
                         activation='sigmoid',
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor), None


class E2VIDRecurrent(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDRecurrent, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        self.unetrecurrent = UNetRecurrent(num_bins=self.num_bins,
                                           num_output_channels=1,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)
        self.prev_states = None

    def reset_states(self):
        self.prev_states = None

    def forward(self, inputs):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        event_tensor = inputs['events']
        img_pred, self.prev_states = self.unetrecurrent.forward(event_tensor, self.prev_states)
        return {'image': img_pred}


class FireNet(nn.Module):
    """
    Refactored version of model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, num_bins=5, base_num_channels=16, kernel_size=3, unet_kwargs=None):
        super().__init__()
        if unet_kwargs:  # legacy compatibility - modern config should not have unet_kwargs
            num_bins = unet_kwargs.get('num_bins', num_bins)
            base_num_channels = unet_kwargs.get('base_num_channels', base_num_channels)
            kernel_size = unet_kwargs.get('kernel_size', kernel_size)
        self.num_bins = num_bins
        padding = kernel_size // 2
        self.head = ConvLayer(self.num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states()

    # @property
    # def states(self):
    #     return copy_states(self._states)
    #
    # @states.setter
    # def states(self, states):
    #     self._states = states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def forward(self, inputs):
        """
        :param x: N x num_input_channels x H x W event tensor
        :return: N x num_output_channels x H x W image
        """
        x = inputs['events']
        x = self.head(x)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        return {'image': self.pred(x)}


class FireNetOrg(BaseE2VID):
    """
    Model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
    The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
    However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
        kernel_size = config.get('kernel_size', 3)
        recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
        self.net = UNetFire(self.num_bins,
                            num_output_channels=1,
                            skip_type=self.skip_type,
                            recurrent_block_type=self.recurrent_block_type,
                            base_num_channels=self.base_num_channels,
                            num_residual_blocks=self.num_residual_blocks,
                            norm=self.norm,
                            kernel_size=kernel_size,
                            recurrent_blocks=recurrent_blocks)
        self.prev_states = None

    def reset_states(self):
        self.prev_states = None

    def forward(self, inputs):
        event_tensor = inputs['events']
        img, self.prev_states = self.net.forward(event_tensor, self.prev_states)
        return {'image': img}


# class FireNetOrg(BaseE2VID):
#     """
#     Model from the paper: "Fast Image Reconstruction with an Event Camera", Scheerlinck et. al., 2019.
#     The model is essentially a lighter version of E2VID, which runs faster (~2-3x faster) and has considerably less parameters (~200x less).
#     However, the reconstructions are not as high quality as E2VID: they suffer from smearing artefacts, and initialization takes longer.
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         self.recurrent_block_type = str(config.get('recurrent_block_type', 'convgru'))
#         recurrent_blocks = config.get('recurrent_blocks', {'resblock': [0]})
#         BN_momentum = config.get('BN_momentum', 0.1)
#         self.net = UNetFire(self.num_bins,
#                             num_output_channels=1,
#                             skip_type=self.skip_type,
#                             recurrent_block_type=self.recurrent_block_type,
#                             base_num_channels=self.base_num_channels,
#                             num_residual_blocks=self.num_residual_blocks,
#                             norm=self.norm,
#                             kernel_size=self.kernel_size,
#                             recurrent_blocks=recurrent_blocks,
#                             BN_momentum=BN_momentum)
#         self.prev_states = None
#
#     def reset_states(self):
#         self.prev_states = None
#
#     def forward(self, inputs):
#         event_tensor = inputs['events']
#         img, self.prev_states = self.net.forward(event_tensor, self.prev_states)
#         return {'image': img}


class FlowNet(BaseModel):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__(unet_kwargs)
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetflow = UNetFlow(unet_kwargs)

    def reset_states(self):
        self.unetflow.states = [None] * self.unetflow.num_encoders

    def forward(self, inputs):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        event_tensor = inputs['events']
        output_dict = self.unetflow.forward(event_tensor)
        return output_dict

