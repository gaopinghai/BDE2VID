import json
import cv2
import sys

import torch
from mmengine.visualization import Visualizer
from mmengine.registry import VISUALIZERS
from mmengine.logging import MessageHub

from utils_func.utils import quick_norm
from utils_func.training_utils import flow2rgb_torch


@VISUALIZERS.register_module()
class E2vidVis(Visualizer):
    def init(self, epochs, iters_per_epoch, num_video_per_epoch=5, totxt=False,
             checkpoint_file=None):
        self.checkpoint_file = checkpoint_file
        self.epochs = epochs
        self.iters_per_epoch = iters_per_epoch
        self.totxt = totxt
        self.video_save_gap = max(iters_per_epoch // num_video_per_epoch, 1)
        self.cur_epoch = None
        self.cur_iter = None
        self.glob_iter = None
        self.message_hub = None
        self.val_seq_save_keys = ['bike', 'outdoor', 'indoor']

    def save_val_vid(self, seq_name):
        for key in self.val_seq_save_keys:
            if key in seq_name:
                save_vid = True
                break
        else:
            save_vid = False
        return save_vid

    def cur_step(self):
        return self.cur_epoch * self.iters_per_epoch + self.cur_iter

    def update_iter(self):
        if self.message_hub is None:
            self.message_hub = MessageHub.get_current_instance()
        self.cur_epoch = self.message_hub.get_info('epoch')
        self.cur_iter = self.message_hub.get_info('iter') % self.iters_per_epoch
        self.glob_iter = self.message_hub.get_info('iter')

    def to_save_vid(self):
        return self.cur_iter % self.video_save_gap == 0

    def add_video(self, name, datas, step, fps=20):
        vid_tensor = make_movie(*datas)
        tb = self.get_backend('TensorboardVisBackend')
        if tb is not None:
            tb.experiment.add_video(name, vid_tensor, global_step=step, fps=fps)

    def savetxt(self, results, loss_name):
        outfile = f"_{loss_name}.txt"
        outfile = self.checkpoint_file.replace('.pth', outfile)
        with open(outfile, 'w') as fp:
            json.dump(results, fp)

    def show_im(self, im):
        cv2.imshow('result', im)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            sys.exit(0)


def make_movie(event_previews, predicted_frames, groundtruth_frames):
    event_previews = torch.cat(event_previews).unsqueeze(0)
    event_previews -= event_previews.min()
    event_previews /= event_previews.max()
    predicted_frames = torch.cat(predicted_frames).unsqueeze(0)
    groundtruth_frames = torch.cat(groundtruth_frames).unsqueeze(0)
    video_tensor = torch.cat([event_previews, predicted_frames, groundtruth_frames], dim=-1)
    return video_tensor
