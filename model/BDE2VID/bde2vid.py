import os
from itertools import chain
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.registry import MODELS

from model.BDE2VID.submodules import ConvLayer, RecurrentConv
from model.BDE2VID.bde2vid_cross_scale_propogation_V5 import BDE2VIDCrossscalePropogationV5


@MODELS.register_module()
class BDE2VID(BaseModel):
    def __init__(self, generator, cpu_cache_length=100, init_cfg=None):
        super(BDE2VID, self).__init__(init_cfg=init_cfg)
        self.cpu_cache_length = cpu_cache_length
        self.generator_cfg = generator
        self.generator = MODELS.build(generator)
        self.recurrentLayers = [m for m in self.generator.modules() if type(m) == RecurrentConv]
        self.vis = None

    def reset_states(self):
        for m in self.recurrentLayers:
            m.state = None
        if hasattr(self.generator, "reset_states"):
            self.generator.reset_states()
        if not self.training:
            self.vis = None

    def forward(self, inputs, mode='tensor', **kwargs):
        self.reset_states()
        if mode == 'loss':

            loss_dict, predicts, event_previews, predicted_frames, groundtruth_frames = \
                self.generator(inputs, record=False, out_preds=False,
                               out_loss=True, cpu_cache_length=self.cpu_cache_length)
            return loss_dict
        elif mode == 'predict':

            seq_folder = kwargs.get('base_folder', 'unknown')[0]
            loss_dict, predicts, event_previews, predicted_frames, groundtruth_frames = \
                self.generator(inputs, record=False,
                               out_preds=True, out_loss=False, cpu_cache_length=self.cpu_cache_length)
            gts = [item['frame'] for item in inputs]
            return predicts, gts
        elif mode == 'tensor':
            loss_dict, predicts, event_previews, predicted_frames, groundtruth_frames = \
                self.generator(inputs, record=False, out_preds=True, out_loss=False,
                               cpu_cache_length=self.cpu_cache_length)
            return predicts