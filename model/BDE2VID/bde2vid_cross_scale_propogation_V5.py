import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from mmengine.model import BaseModule
from mmengine.registry import MODELS
from model.BDE2VID.activaions import ACTIVATION

from model.BDE2VID.submodules import ResidualBlock, UpsampleConvLayer, ConvLayer
from model.BDE2VID.DTransformer import DFrameAttention
from model.losses.losses import LOSSES
from model.BDE2VID.submodules import ConvLayer, RecurrentConv



@MODELS.register_module()
class BDE2VIDCrossscalePropogationV5(BaseModule):
    def __init__(self, num_bins, basechannels, num_encoders, ks, num_res_blocks, norm=None,
                 recurrent_block_type='convlstm', useRC=True, skip_type='sum', activation=None,
                 num_output_channels=1, act_net="default",  buffer_index=None, q_idx=None,
                 window_size=(7, 7), nwindow_size=None, depths=[4, 0, 6], num_heads=16, drop_path_rate=0.2,
                 use_checkpoint=False, act_attn="default", losses=None, loss_inds=None, init_cfg=None):
        super(BDE2VIDCrossscalePropogationV5, self).__init__(init_cfg=init_cfg)
        if activation is None:
            activation = dict(type='Sigmoid')
        self.skip_type = skip_type
        if self.skip_type == 'sum':
            self.apply_skip_connection = skip_sum
        elif self.skip_type == 'concat':
            self.apply_skip_connection = skip_concat
        elif self.skip_type == 'no_skip' or self.skip_type is None:
            self.apply_skip_connection = nn.Identity()
        else:
            raise KeyError('Could not identify skip_type, please add "skip_type":'
                           ' "sum", "concat" or "no_skip" to config["model"]')
        self.losses_cfg = losses
        self.losses = {loss['type']: LOSSES.build(loss) for loss in losses}
        if loss_inds is not None and type(loss_inds) is not list:
            loss_inds = [tmp for tmp in range(40) if tmp % loss_inds == 0]
        self.loss_inds = loss_inds

        self.head = ConvLayer(num_bins, basechannels, kernel_size=ks, stride=1, padding=ks // 2,
                              norm=norm, activation=act_net)
        self.forward_encoder = Encoder(basechannels, num_encoders, ks, num_res_blocks,
                                       norm, useRC=useRC, recurrent_block_type=recurrent_block_type,
                                       activation=act_net)
        self.backward_encoder = Encoder(basechannels, num_encoders, ks, num_res_blocks,
                                       norm, useRC=useRC, recurrent_block_type=recurrent_block_type,
                                       activation=act_net)

        encoder_in_channels = [basechannels * 2 ** i for i in range(num_encoders)]
        encoder_out_channels = [basechannels * 2 ** (i + 1) for i in range(num_encoders)]
        self.fusion_layers = nn.ModuleList()
        for en_out_chns in encoder_out_channels:
            fusion_layer = nn.Conv2d(en_out_chns*2, en_out_chns, 1, 1, 0, bias=True)
            self.fusion_layers.append(fusion_layer)

        decoder_ins = [basechannels * 2 ** (i + 1) for i in range(num_encoders)][::-1]
        decoder_outs = [basechannels * 2 ** i for i in range(num_encoders)][::-1]
        maxchannels = decoder_ins[0]

        frame_num = len(buffer_index)
        self.q_idx = q_idx
        self.buffer_index = np.array(buffer_index)
        self.feat_attns = nn.ModuleList()
        for depth, chns in zip(depths, encoder_out_channels):
            if depth > 0:
                dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
                self.feat_attns.append(DFrameAttention(
                    chns, depth, num_heads, (frame_num, *window_size),
                    nwindow_size=nwindow_size, q_ind=q_idx, drop_path=dpr,
                    use_checkpoint=use_checkpoint, activation=act_attn)
                )
            else:
                self.feat_attns.append(None)
        if self.feat_attns[-1] is None:
            self.feat_attns[-1] = nn.Sequential(
                ParseLayer(),
                *[ResidualBlockNoBN(maxchannels) for _ in range(num_res_blocks)])

        self.decoders = nn.ModuleList()
        for in_chns, out_chns in zip(decoder_ins, decoder_outs):
            upsampleLayer = UpsampleConvLayer(in_chns, out_chns, ks, padding=ks // 2,
                                              norm=norm, activation='ReLU6')
            if skip_type == 'concat':
                fusion_layer = nn.Conv2d(in_chns * 2, in_chns, 1, 1, 0, bias=True)
            else:
                fusion_layer = nn.Identity()
            self.decoders.append(nn.Sequential(fusion_layer, upsampleLayer))

        if skip_type == 'concat':
            fusion_layer = nn.Conv2d(basechannels * 2, basechannels, 1, 1, 0, bias=True)
        else:
            fusion_layer = nn.Identity()
        self.predI = nn.Sequential(fusion_layer,
                                   nn.Conv2d(basechannels, num_output_channels, 1, 1, 0, bias=True))
        self.activation = ACTIVATION.build(activation)

    def forward(self, input_seqs, record, out_preds, out_loss, cpu_cache_length):
        T = len(input_seqs)
        self.cpu_cache = True if T > cpu_cache_length else False
        if record:
            event_previews = [None] * T
            predicted_frames = [None] * T  # list of intermediate predicted frames
            groundtruth_frames = [None] * T
        else:
            event_previews = None
            predicted_frames = None
            groundtruth_frames = None
        if out_preds:
            predicts = [None] * T
        else:
            predicts = None

        head_seqs = [self.head(d['events']).cpu() if self.cpu_cache else self.head(d['events']) for d in input_seqs]
        mearged_feats_all_lvl = []
        target_seqs = head_seqs
        for ind_l in range(len(self.forward_encoder)):
            forward_feat_seqs = [None]*T
            backward_feat_seqs = [None]*T
            for idx_f in range(T):
                idx_b = T - 1 - idx_f
                head_f = target_seqs[idx_f]
                head_b = target_seqs[idx_b]
                if self.cpu_cache:
                    head_f = head_f.cuda()
                    head_b = head_b.cuda()
                forward_feat = self.forward_encoder[ind_l](head_f)
                backward_feat = self.backward_encoder[ind_l](head_b)
                if self.cpu_cache:
                    forward_feat = forward_feat.cpu()
                    backward_feat = backward_feat.cpu()
                forward_feat_seqs[idx_f] = forward_feat
                backward_feat_seqs[idx_b] = backward_feat

            merged_feat_seqs = [None]*T
            for idx_i in range(T):
                ff = forward_feat_seqs[idx_i]
                fb = backward_feat_seqs[idx_i]
                if self.cpu_cache:
                    ff = ff.cuda()
                    fb = fb.cuda()
                merged_feat = ff + fb
                if self.cpu_cache:
                    merged_feat = merged_feat.cpu()
                merged_feat_seqs[idx_i] = merged_feat

            if ind_l == len(self.forward_encoder) - 1:
                mearged_feats_all_lvl.append(merged_feat_seqs)
            if self.feat_attns[ind_l] is not None:
                empty_x = torch.zeros_like(merged_feat_seqs[0], requires_grad=False)
                empty_x = empty_x.cuda()
                for idx_t in range(T):
                    feats_buffer = []
                    for i in self.buffer_index:
                        idx = i + idx_t
                        if idx >= 0 and idx < T:
                            feat = merged_feat_seqs[idx]
                        else:
                            feat = empty_x
                        if self.cpu_cache:
                            feat = feat.cuda()
                        feats_buffer.append(feat)
                    x = self.feat_attns[ind_l](feats_buffer)
                    x = x + merged_feat_seqs[idx_t].cuda()
                    if self.cpu_cache:
                        x = x.cpu()
                    merged_feat_seqs[idx_t] = x

            target_seqs = merged_feat_seqs
            mearged_feats_all_lvl.append(merged_feat_seqs)

        losses = defaultdict(list)
        prev_img_gt = None
        prev_img_pd = None
        for k, func in self.losses.items():
            try:
                func.reset()
            except:
                pass

        for idx_t in range(T):
            head = head_seqs[idx_t]
            x = mearged_feats_all_lvl[-1][idx_t]
            if self.cpu_cache:
                head = head.cuda()
                x = x.cuda()

            for i, decoder in enumerate(self.decoders):
                feat_prev_lvl = mearged_feats_all_lvl[-2-i][idx_t]
                if self.cpu_cache:
                    feat_prev_lvl = feat_prev_lvl.cuda()
                x = decoder(self.apply_skip_connection([feat_prev_lvl, x]))
            x = self.apply_skip_connection([x, head])
            img = self.predI(x)
            img = self.activation(img)

            if out_preds:
                with torch.no_grad():
                    # predicts.append(img)
                    predicts[idx_t] = img
            if record:
                with torch.no_grad():
                    new_events, new_frame = input_seqs[idx_t]['events'], input_seqs[idx_t]['frame']
                    event_previews[idx_t] = torch.sum(new_events[0:1], dim=1).unsqueeze(1)
                    predicted_frames[idx_t] = img[0:1].clone()
                    groundtruth_frames[idx_t] = new_frame[0:1].clone()

            if out_loss and (self.loss_inds is None or idx_t in self.loss_inds):
                cur_img_gt = input_seqs[idx_t]['frame']
                cur_img_pd = img
                for k, func in self.losses.items():
                    if k in ['PerceptualLoss', 'L1Loss']:
                        losses[k].append(func(cur_img_pd, cur_img_gt))
                    elif k in ['VIPLoss']:
                        losses[k].append(func(cur_img_pd, cur_img_gt, prev_img_gt, prev_img_pd))
                    elif k in ['TemporalConsistencyLoss']:
                        if idx_t > func.L0:
                            losses[k].append(func(prev_img_gt, cur_img_gt, prev_img_pd, cur_img_pd,
                                                  input_seqs[idx_t]['flow']))
                    else:
                        raise ValueError(f"Unknown loss {k} in forward model.!")
            if out_loss:
                prev_img_gt, prev_img_pd = input_seqs[idx_t]['frame'], img

        if out_loss:
            loss_dict = {}
            loss = None
            for k, v in losses.items():
                v = sum(v) / len(v)
                loss = v if loss is None else loss+v
                with torch.no_grad():
                    loss_dict[f"L_{''.join([c for c in k if c.isupper()])}"] = v

            with torch.no_grad():
                loss_dict['loss'] = loss
        else:
            loss_dict = None

        return loss_dict, predicts, event_previews, predicted_frames, groundtruth_frames


def Encoder(basechannels, num_encoders, ks, num_res_blocks, norm, useRC=False,
            recurrent_block_type='convlstm', activation="default"):
        activation = "ReLU" if activation == "default" else activation
        encoder_in_channels = [basechannels * 2**i for i in range(num_encoders)]
        encoder_out_channels = [basechannels * 2**(i+1) for i in range(num_encoders)]
        encoders = nn.ModuleList()
        for in_chns, out_chns in zip(encoder_in_channels, encoder_out_channels):
            if useRC:
                encoder = RecurrentConv(in_chns, out_chns, kernel_size=ks, stride=2,
                                        padding=ks//2, norm=norm, activation=activation,
                                        recurrent_block_type=recurrent_block_type)
            else:
                encoder = ConvLayer(in_chns, out_chns, kernel_size=ks, stride=2,
                                    padding=ks//2, norm=norm, activation=activation)
            encoders.append(encoder)
        return encoders


class ResidualBlockNoBN(BaseModule):
    def __init__(self, mid_channels=64, activation="default", init_cfg=None):
        super().__init__(init_cfg)
        activation = "ReLU" if activation == "default" else activation
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.activation = getattr(torch.nn, activation)(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv2(self.activation(self.conv1(x)))
        return residual + out


class ParseLayer(nn.Module):
    def __init__(self):
        super(ParseLayer, self).__init__()

    def forward(self, x):
        return x[0]


def skip_concat(xs):
    return torch.cat(xs, dim=1)


def skip_sum(xs):
    x = xs[0]
    for d in xs[1:]:
        x = x + d
    return x