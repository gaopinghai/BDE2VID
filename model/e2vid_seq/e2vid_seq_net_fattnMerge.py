import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
# from pytorch_memlab import profile, profile_every

from mmengine.registry import MODELS
from model.activaions import ACTIVATION

from utils_func.inference_utils import Croper
from model.submodules import ConvLayer, SepDconv, RecurrentConv, UpsampleConvLayer, \
    PixelShufflePack
from model.losses.losses import LOSSES
from model.e2vid_seq.e2vid_seq_net import OrgBackward, Encoder, ResidualBlockNoBN, FusionLayer, \
    skip_sum, skip_concat, identity, E2VIDSeqNet
from model.e2vid_seq.e2vid_seq_net_fattn import MutilevelFrameAttn, MutilevelFrameAttnWithPM
from model.e2vid_seq.transformer import FrameAttention, PatchMerging


@MODELS.register_module()
class E2VIDSeqNetFAMerge(E2VIDSeqNet):
    pass



@MODELS.register_module()
class OrgForwardFAMerge(nn.Module):
    def __init__(self, basechannels, num_encoders, num_res_blocks, ks, norm=None, useRC=False,
                 recurrent_block_type='convlstm', skip_type='sum', num_output_channels=1,
                 activation=dict(type='Sigmoid'), fusion_type="conv", upsample_type="UpsampleConv",
                 losses=[dict(type="PerceptualLoss", net='alex')],
                 MutiFrameAttnType="MutilevelFrameAttn", act_net="default",
                 depths=[2, 2, 2, 2], num_heads=[4, 8, 16, 32], window_size=(3, 7, 7),
                 window_dilate=1, drop_path_rate=0.2, use_checkpoint=False):
        super().__init__()
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
        self.losses_cfg = losses
        self.losses = {loss['type']: LOSSES.build(loss) for loss in losses}

        self.encoder = Encoder(basechannels, num_encoders, ks, norm, useRC,
                               recurrent_block_type, activation=act_net)
        decoder_ins = self.encoder.encoder_out_channels[::-1]
        decoder_outs = self.encoder.encoder_in_channels[::-1]
        maxchannels = decoder_ins[0]
        self.resblocks = nn.Sequential(
            *[ResidualBlockNoBN(maxchannels, activation=act_net) for i in range(num_res_blocks)]
        )

        if MutiFrameAttnType == "MutilevelFrameAttn":
            MutiFrameAttn = MutilevelFrameAttn
        elif MutiFrameAttnType == "MutilevelFrameAttnWithPM":
            MutiFrameAttn = MutilevelFrameAttnWithPM
        else:
            raise ValueError(f"unknown MutiFrameAttnType: {MutiFrameAttnType}")
        self.num_fattn_layers = len(depths)
        feat_chns = [basechannels * 2**i for i in range(num_encoders+1)]
        feat_chns = feat_chns[-self.num_fattn_layers:]
        self.fattn = MutiFrameAttn(feat_chns, depths, num_heads, window_size,
                                        drop_path_rate, use_checkpoint=use_checkpoint)
        self.num_frame = window_size[0]
        self.window_dilate = window_dilate
        self.num_size = self.num_frame//2 * (window_dilate+1)
        self.frame_inds = (np.arange(self.num_frame) - self.num_frame // 2) * (window_dilate + 1)
        self.frame_inds_unit = np.linspace(-1, 1, self.num_frame)

        self.fusioners = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for in_chns, out_chns in zip(decoder_ins, decoder_outs):
            if upsample_type == "UpsampleConv":
                upsampleLayer = UpsampleConvLayer(in_chns, out_chns, ks, padding=ks//2,
                                                  norm=norm, activation=act_net)
            elif upsample_type == "PixelShuffle":
                upsampleLayer = PixelShufflePack(in_chns, out_chns, scale_factor=2,
                                                 upsample_kernel=ks)
            elif upsample_type == "RecurrentUpConv":
                upsampleLayer = nn.Sequential(
                    RecurrentConv(in_chns, out_chns, kernel_size=ks, padding=ks//2,
                                  recurrent_block_type=recurrent_block_type,
                                  norm=norm, activation=act_net),
                    UpsampleConvLayer(out_chns, out_chns, kernel_size=ks,
                                      padding=ks//2, norm=norm, activation=act_net)
                )
            else:
                raise ValueError(f"unknown upsample_type: {upsample_type}")
            self.decoders.append(upsampleLayer)
            if fusion_type == 'conv':
                fusion_layer = nn.Conv2d(in_chns*2, in_chns, 1, 1, 0, bias=True)
            elif fusion_type == 'sum':
                fusion_layer = FusionLayer()
            else:
                raise ValueError(f"Unknown fusion_type {fusion_type}")
            self.fusioners.append(fusion_layer)

        self.predI = nn.Conv2d(basechannels, num_output_channels, 1, 1, 0, bias=True)
        self.activation = ACTIVATION.build(activation)

    def forward(self, input_seqs, head_seqs, backward_feat_seqs, record=False, out_preds=False,
                out_loss=False, cpu_cache_length=100):
        T = len(head_seqs)
        if record:
            event_previews = [None]*T
            predicted_frames = [None]*T  # list of intermediate predicted frames
            groundtruth_frames = [None]*T
        else:
            event_previews = None
            predicted_frames = None
            groundtruth_frames = None
        if out_preds:
            predicts = [None]*T
        else:
            predicts = None
        losses = defaultdict(list)
        prev_img_gt = None
        prev_img_pd = None

        backward_feat_seqs = [[head, *bfeats] for head, bfeats in zip(head_seqs, backward_feat_seqs)]
        # T = len(head_seqs)
        merged_feat_seqs = [None]*T
        self.cpu_cache = True if T > cpu_cache_length else False
        mid_num = self.num_frame // 2
        for ind_t in range(T):
            feats_backward = backward_feat_seqs[ind_t]
            if self.cpu_cache:
                feats_backward = [f.cuda() for f in feats_backward]
            head = feats_backward[0]
            feats_forward = self.encoder(head)
            feats_cur = [fusion(torch.cat([ff, fb], dim=1)) for fusion, ff, fb in
                     zip(self.fusioners[::-1], feats_forward, feats_backward[1:])]
            feats_cur = [head, *feats_cur]

            numR = T - 1 - ind_t
            if ind_t >= mid_num and numR >= mid_num:
                frame_ind = self.frame_inds + ind_t
                frame_ind_min = (self.frame_inds_unit * ind_t + ind_t).astype(int)
                frame_ind[:mid_num] = np.maximum(frame_ind, frame_ind_min)[:mid_num]
                frame_ind_max = (self.frame_inds_unit * numR + ind_t).astype(int)
                frame_ind[mid_num:] = np.minimum(frame_ind, frame_ind_max)[mid_num:]
                frame_ind_left, frame_ind_right = frame_ind[:mid_num], frame_ind[mid_num+1:]
                feats_left = [merged_feat_seqs[tmp_i] for tmp_i in frame_ind_left]
                feats_right = [backward_feat_seqs[tmp_i] for tmp_i in frame_ind_right]
                feats_window = [*feats_left, feats_cur, *feats_right]
                if self.cpu_cache:
                    feats_window = [[f.cuda() for f in feats] for feats in feats_window]
                feats = self.fattn(feats_window)
            else:
                feats = feats_cur
            if self.cpu_cache:
                feats_cpu = [f.cpu() for f in feats]
            else:
                feats_cpu = feats
            merged_feat_seqs[ind_t] = feats_cpu

            x = feats[-1]
            x = self.resblocks(x)
            for i, decoder in enumerate(self.decoders):
                x = decoder(self.apply_skip_connection([feats[-1-i], x]))
            head = feats[0]
            x = self.apply_skip_connection([x, head])
            img = self.predI(x)
            img = self.activation(img)

            if out_preds:
                with torch.no_grad():
                    # predicts.append(img)
                    predicts[ind_t] = img
            if record:
                with torch.no_grad():
                    new_events, new_frame = input_seqs[ind_t]['events'], input_seqs[ind_t]['frame']
                    # event_previews.append(torch.sum(new_events[0:1], dim=1).unsqueeze(1))
                    # predicted_frames.append(img[0:1].clone())
                    # groundtruth_frames.append(new_frame[0:1].clone())
                    event_previews[ind_t] = torch.sum(new_events[0:1], dim=1).unsqueeze(1)
                    predicted_frames[ind_t] = img[0:1].clone()
                    groundtruth_frames[ind_t] = new_frame[0:1].clone()

            if out_loss:
                cur_img_gt = input_seqs[ind_t]['frame']
                cur_img_pd = img
                for k, func in self.losses.items():
                    if k in ['PerceptualLoss', 'L1Loss']:
                        losses[k].append(func(cur_img_pd, cur_img_gt))
                    elif k == 'TemporalConsistencyLoss':
                        if ind_t >= func.L0:
                            losses[k].append(func(prev_img_gt, cur_img_gt, prev_img_pd, cur_img_pd,
                                                  input_seqs[ind_t]['flow']))
                    else:
                        raise ValueError(f"Unknown loss {k} in forward model.!")
                prev_img_gt, prev_img_pd = cur_img_gt, cur_img_pd

        if out_loss:
            loss_dict = {}
            loss = None
            for k, v in losses.items():
                v = sum(v) / len(v)
                loss = v if loss is None else loss+v
                with torch.no_grad():
                    loss_dict[f"L_{k[0]}"] = v

            with torch.no_grad():
                loss_dict['loss'] = loss
        else:
            loss_dict = None

        return loss_dict, predicts, event_previews, predicted_frames, groundtruth_frames


