import os
import torch
from collections import defaultdict

from mmengine.model import BaseModel
from mmengine.registry import MODELS

from model.e2vid_seq.e2vid_seq_net import E2VIDSeqNet
from model.e2vid_seq.e2vid_seq_net_fattn import E2VIDSeqNetFattn
from model.e2vid_seq.e2vid_seq_net_fattnMerge import E2VIDSeqNetFAMerge
from model.losses.losses import LOSSES
from model.submodules import RecurrentConv, RecurrentUpConv
from visulize.e2vid_vis import E2vidVis


@MODELS.register_module()
class E2VIDSeq(BaseModel):
    def __init__(self, generator):
        """
        Args:
            generator (dict): config for submodel.
            losses (list): each item is config for one loss function.
        """
        super(E2VIDSeq, self).__init__()
        self.generator_cfg = generator
        self.generator = MODELS.build(generator)
        self.recurrentLayers = [m for m in self.generator.modules() if type(m) == RecurrentConv or type(m) == RecurrentUpConv]
        self.vis = None

    def forward(self, inputs, mode='tensor', **kwargs):
        self.reset_states()

        ################# loss mode #####################
        if mode == 'loss':
            if self.vis is None:
                self.vis = E2vidVis.get_current_instance()
            self.vis.update_iter()

            loss_dict, predicts, event_previews, predicted_frames, gt_frames \
                = self.generator(inputs, self.vis.to_save_vid(), out_preds=False,
                                 out_loss=True)
            if self.vis.to_save_vid():
                self.vis.add_video(f"movie_train_{self.vis.cur_iter}",
                                   [event_previews, predicted_frames, gt_frames],
                                   self.vis.cur_epoch)
            return loss_dict
        ################# predict mode #####################
        elif mode == 'predict':
            self.vis.update_iter()

            seq_folder = kwargs.get('base_folder', 'unknown')[0]
            loss_dict, predicts, event_previews, predicted_frames, gt_frames \
                = self.generator(inputs, self.vis.save_val_vid(seq_folder), out_preds=True,
                                 out_loss=False)
            if self.vis.save_val_vid(seq_folder):
                self.vis.add_video(f"movie_val_{seq_folder.split(os.sep)[-1]}",
                                   [event_previews, predicted_frames, gt_frames],
                                   self.vis.cur_epoch)
            gts = [item['frame'] for item in inputs]
            return predicts, gts
        ################# tensor mode #####################
        elif mode == 'tensor':
            loss_dict, predicts, event_previews, predicted_frames, gt_frames \
                = self.generator(inputs, record=False, out_preds=True, out_loss=False)
            return predicts

        elif mode == 'loss_check':
            loss_dict, predicts, event_previews, predicted_frames, gt_frames \
                = self.generator(inputs, record=False, out_preds=False, out_loss=True)
            return loss_dict


    def reset_states(self):
        for m in self.recurrentLayers:
            m.state = None