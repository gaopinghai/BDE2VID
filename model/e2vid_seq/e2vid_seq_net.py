import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from mmengine.registry import MODELS
from model.activaions import ACTIVATION

from utils_func.inference_utils import Croper
from model.submodules import ConvLayer, SepDconv, RecurrentConv, UpsampleConvLayer, \
    PixelShufflePack
from model.losses.losses import LOSSES


@MODELS.register_module()
class E2VIDSeqNet(nn.Module):
    def __init__(self, backward_module, forward_module, direction='bidirection'):
        """
        Args:
            backward_module (dict): dict(type="OrgBackward", num_bins=5, basechannels=32,
                                            num_encoders=3, ks=5, norm=None, useRC=False,
                                            recurrent_block_type='convlstm')
            forwar_module (dict): dict(type="OrgForward", basechannels=32, num_encoders=3,
                                    num_res_blocks=2, ks=5, norm=None, useRC=False,
                                    recurrent_block_type='convlstm', skip_type='sum',
                                    activation=dict(type='Sigmoid'), num_output_channels=1,
                                    losses=[dict(type="PerceptualLoss", net='alex')])
        """
        super().__init__()
        self.backward_module_cfg = backward_module
        self.backward_module = MODELS.build(backward_module)
        self.forward_module_cfg = forward_module
        self.forward_module = MODELS.build(forward_module)
        self.direction = direction

    def forward(self, input_seqs, record=False, out_preds=True, out_loss=False, cpu_cache_length=100):
        """Forward function for E2VIDSeqNet
        Args:
            input_seqs (list): input list with L item,
            each item is a dict

        Returns:
            predicts (list): output list with L item,
            each item is a dict
        """

        # backward propagation
        head_seqs, backward_feat_seqs = self.backward_module(input_seqs, cpu_cache_length,
                                                             self.direction)
        loss_dict, predicts, event_previews, predicted_frames, groundtruth_frames = \
            self.forward_module(input_seqs, head_seqs, backward_feat_seqs,
                                record, out_preds, out_loss, cpu_cache_length,
                                self.direction)

        return loss_dict, predicts, event_previews, predicted_frames, groundtruth_frames


@MODELS.register_module()
class OrgForward(nn.Module):
    def __init__(self, basechannels, num_encoders, num_res_blocks, ks, norm=None, useRC=False,
                 recurrent_block_type='convlstm', skip_type='sum', num_output_channels=1,
                 activation=dict(type='Sigmoid'), fusion_type="conv", upsample_type="UpsampleConv",
                 losses=[dict(type="PerceptualLoss", net='alex')], act_net="default"):
        super(OrgForward, self).__init__()
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
            *[ResidualBlockNoBN(maxchannels, activation=act_net) for _ in range(num_res_blocks)]
        )

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
                    RecurrentConv(in_chns, out_chns, kernel_size=ks, padding=ks//2, norm=norm,
                                  recurrent_block_type=recurrent_block_type,
                                  activation=act_net),
                    UpsampleConvLayer(out_chns, out_chns, kernel_size=ks, padding=ks//2,
                                      norm=norm, activation=act_net)
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

        # T = len(head_seqs)
        self.cpu_cache = True if T > cpu_cache_length else False

        for t in range(T):
            head = head_seqs[t]
            feats_backward = backward_feat_seqs[t]
            if self.cpu_cache:
                head = head.cuda()
                feats_backward = [f.cuda() for f in feats_backward]

            feats = self.encoder(head)
            x = torch.cat([feats[-1], feats_backward[-1]], dim=1)
            x = self.fusioners[0](x)
            merged_x = x
            x = self.resblocks(x)

            for i, decoder in enumerate(self.decoders):
                if i > 0:
                    merged_x = torch.cat([feats[-1-i], feats_backward[-1-i]], dim=1)
                    merged_x = self.fusioners[i](merged_x)
                x = decoder(self.apply_skip_connection([merged_x, x]))

            x = self.apply_skip_connection([x, head])
            img = self.predI(x)
            img = self.activation(img)

            if out_preds:
                with torch.no_grad():
                    # predicts.append(img)
                    predicts[t] = img
            if record:
                with torch.no_grad():
                    new_events, new_frame = input_seqs[t]['events'], input_seqs[t]['frame']
                    # event_previews.append(torch.sum(new_events[0:1], dim=1).unsqueeze(1))
                    event_previews[t] = torch.sum(new_events[0:1], dim=1).unsqueeze(1)
                    # predicted_frames.append(img[0:1].clone())
                    predicted_frames[t] = img[0:1].clone()
                    # groundtruth_frames.append(new_frame[0:1].clone())
                    groundtruth_frames[t] = new_frame[0:1].clone()

            if out_loss:
                cur_img_gt = input_seqs[t]['frame']
                cur_img_pd = img
                for k, func in self.losses.items():
                    if k in ['PerceptualLoss', 'L1Loss']:
                        losses[k].append(func(cur_img_pd, cur_img_gt))
                    elif k == 'TemporalConsistencyLoss':
                        if t >= func.L0:
                            losses[k].append(func(prev_img_gt, cur_img_gt, prev_img_pd, cur_img_pd,
                                                  input_seqs[t]['flow']))
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


@MODELS.register_module()
class OrgBackward(nn.Module):
    def __init__(self, num_bins, basechannels, num_encoders, ks, norm=None, useRC=False,
                 recurrent_block_type='convlstm', act_net="default"):
        super(OrgBackward, self).__init__()
        self.head = ConvLayer(num_bins, basechannels, kernel_size=ks, stride=1, padding=ks//2,
                              norm=norm, activation=act_net)
        self.encoder = Encoder(basechannels, num_encoders, ks, norm, useRC,
                               recurrent_block_type, activation=act_net)

    def forward(self, input_seqs, cpu_cache_length=100, direction="bidirection"):
        t = len(input_seqs)
        head_seqs = [None]*t
        backward_feat_seqs = [None]*t
        self.cpu_cache = True if t > cpu_cache_length else False

        for i in range(t-1, -1, -1):
            events = input_seqs[i]['events']
            feat = self.head(events)

            if direction in ["bidirection", "backward"]:
                feats = self.encoder(feat)
                if self.cpu_cache:
                    feat = feat.cpu()
                    feats = [f.cpu() for f in feats]
            elif direction == "forward":
                feats = None
            else:
                raise ValueError(f"unknown direction: {direction}")

            head_seqs[i] = feat
            backward_feat_seqs[i] = feats
        # backward_feat_seqs = backward_feat_seqs[::-1]
        # head_seqs = head_seqs[::-1]
        return head_seqs, backward_feat_seqs


class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 + x2


class Encoder(nn.Module):
    def __init__(self, basechannels, num_encoders, ks, norm, useRC=False,
                 recurrent_block_type='convlstm', activation="default"):
        super(Encoder, self).__init__()
        activation = "ReLU" if activation == "default" else activation
        self.encoder_in_channels = [basechannels * 2**i for i in range(num_encoders)]
        self.encoder_out_channels = [basechannels * 2**(i+1) for i in range(num_encoders)]
        self.encoders = nn.ModuleList()
        for in_chns, out_chns in zip(self.encoder_in_channels, self.encoder_out_channels):
            if useRC:
                encoder = RecurrentConv(in_chns, out_chns, kernel_size=ks, stride=2,
                                        padding=ks//2, norm=norm, activation=activation,
                                        recurrent_block_type=recurrent_block_type)
            else:
                encoder = ConvLayer(in_chns, out_chns, kernel_size=ks, stride=2,
                                    padding=ks//2, norm=norm, activation=activation)
            self.encoders.append(encoder)

    def forward(self, x):
        blocks = []
        for encoder in self.encoders:
            x = encoder(x)
            blocks.append(x)
        return blocks


class DeformableAlign(nn.Module):
    def __init__(self, offset_inchns, offset_outchns, in_channels, out_channels,
                 kernel_size, padding, stride, num_blocks):
        super(DeformableAlign, self).__init__()

        self.offset_module = ResidualBlocks(offset_inchns, offset_outchns, num_blocks)
        self.deformableAlign = SepDconv(offset_inchns=offset_outchns, in_channels=in_channels,
                                        out_channels=out_channels, kernel_size=kernel_size,
                                        padding=padding, stride=stride)

    def forward(self, offset_x, x, init_offset):
        offset_x = self.offset_module(offset_x)
        return self.deformableAlign(offset_x, x, init_offset)


class ResidualBlocks(nn.Module):
    def __init__(self,in_channels, out_channels, num_blocks):
        super(ResidualBlocks, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        residual_blocks = [ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]
        self.residual_blocks = nn.Sequential(*residual_blocks)

    def forward(self, feat):
        feat = self.head(feat)
        return self.residual_blocks(feat)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64, activation="default"):
        super().__init__()
        activation = "ReLU" if activation == "default" else activation
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.activation = getattr(torch.nn, activation)(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv2(self.activation(self.conv1(x)))
        return residual + out


def skip_concat(xs):
    return torch.cat(xs, dim=1)


def skip_sum(xs):
    x = xs[0]
    for d in xs[1:]:
        x = x + d
    return x


def identity(xs):
    return xs