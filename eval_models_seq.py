import functools
import time
import cv2
if __name__ == '__main__':
    cv2.namedWindow('result')
    cv2.destroyAllWindows()
import os, sys
import platform
from os.path import join
import numpy as np
import argparse
from glob import glob
import copy
import collections
import json
import re
from tqdm import tqdm
import more_itertools

import torch
from mmengine.config import Config
from mmengine.registry import MODELS

from model.spade_e2vid.spade_e2v import SPADEE2VID
from model.BDE2VID.bde2vid import BDE2VID
from model.e2vid.model import FlowNet, E2VIDRecurrent, FireNet, FireNetOrg
from model.EVSNN.model.snn_network import EVSNN_LIF_final, PAEVSNN_LIF_AMPLIF_final
from model.eitr.eitr import EITR
from model.EVSNN.rec_snn_forward import RecSNN
from model.EVSNN.utils.util import normalize_image
from evaluate.metrics import mse_loss, structural_similarity, perceptual_loss
from scripts.generate_table import generate_table
from data_loader.h5_dataset import InferenceDataLoader
from utils_func.inference_utils import Croper, torch2cv2, cv2torch
from utils_func.utils import quick_norm, robust_1_99


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpointfile, args=None, **kwargs):
    """用于加载各种模型的总函数"""
    checkpoint = torch.load(checkpointfile)
    model = None

    if checkpointfile.endswith('firenet_1000.pth'):
        model_args = checkpoint['config']['model']
        model = FireNetOrg(model_args)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    if 'state_dict' in checkpoint:
        if 'meta' in checkpoint:
            model_cfg = checkpoint['meta']['cfg']
            model_cfg = Config.fromstring(model_cfg, '.py')
            model_cfg = model_cfg.model
            model_type = model_cfg['type']
            if model_type.startswith('BDE2VID'):
                args.seq_model = True
            model = MODELS.build(model_cfg)
        elif "arch" in checkpoint:
            # load E2VID
            arch = checkpoint['arch']
            model_args = None
            if 'config' in checkpoint:
                cp = checkpoint['config']
                if type(cp) is not dict:
                    cp = cp.config

                if 'arch' in cp:
                    model_args = cp['arch']['args']
                    if 'unet_kwargs' in model_args:
                        model_args = model_args['unet_kwargs']
                    elif 'eitr_kwargs' in model_args:
                        model_args = model_args['eitr_kwargs']
            elif 'model' in checkpoint:
                model_args = checkpoint['model']
                try:
                    args.normalize = True
                except:
                    pass
            if arch == 'FireNet':
                model = eval(arch)(**model_args)
            else:
                model = eval(arch)(model_args)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        if "SPADE" in checkpointfile:
            model = SPADEE2VID()
            model.load_state_dict(checkpoint)
            args.normalize = True
        elif "SNN" in checkpointfile:
            model_name = "PAEVSNN_LIF_AMPLIF_final" if "PAEVSNN" in checkpointfile \
                else "EVSNN_LIF_final"
            model = RecSNN(model_name, checkpointfile)
    return model


def eval_model_alldata(datafiles, checkpoint_file, args):
    args.checkpoint_name = os.path.split(args.checkpoint_path)[-1].split('.')[0]
    if args.pause_st is not None:
        result_file = f"{args.checkpoint_name}_{args.datatype}_{args.pause_st}_{args.pause_ed}.txt"
        args.subseq_L = None
    elif args.subseq_L is not None:
        result_file = f"{args.checkpoint_name}_L{args.subseq_L}_{args.datatype}.txt"
    else:
        result_file = f"{args.checkpoint_name}_{args.datatype}.txt"
    checkpoint_dir = f"{args.checkpoint_name}_{args.datatype}"
    result_file = os.path.join(args.checkpoint_dir, result_file)
    if os.path.exists(result_file):
        print(f"skiping {checkpoint_file}")
        return

    print('Loading checkpoint: {} ...'.format(checkpoint_file))
    model = load_model(checkpoint_file, args)
    model.eval()
    model.to(device)
    # 在此处生成结果文件防止其他程序重复处理该模型权重
    if os.path.exists(result_file):
        print(f"skiping {checkpoint_file}")
        return
    fp = open(result_file, 'w')
    results = collections.defaultdict(lambda: collections.defaultdict(dict))
    detail_results = collections.defaultdict(lambda: collections.defaultdict(dict))
    pbar = tqdm(datafiles)
    for datafile in pbar:
        pbar.set_description(desc=f"processing {datafile}...")
        args.events_file_path = os.path.join(dataDir, datafile)
        datasetName, filename = datafile.split(os.sep)
        filename = filename.split('.h5')[0]
        args.output_folder = os.path.join(args.output_folder_root, checkpoint_dir, datasetName, filename)
        with torch.no_grad():
            result, detail_res = eval_model(args, model)
        results[datasetName][filename] = result
        detail_results[datasetName][filename] = detail_res

    json.dump(results, fp)
    fp.close()
    print(f"results writed to {result_file}")
    generate_table([result_file], result_file.replace('.txt', '_table.txt'))
    detail_resfile = result_file.replace('.txt', '_detail.txt')
    fp = open(detail_resfile, 'w')
    json.dump(detail_results, fp)
    fp.close()


def eval_model(args, model):
    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 5,
                      'filter_hot_events': args.filter_hot_events,
                      'voxel_method': {'method': 'between_frames',
                                       'k': 0,
                                       't': 0,
                                       'sliding_window_w': 0,
                                       'sliding_window_t': 0},
                      }
    if args.normalize:
        print('Using legacy voxel normalization')
        dataset_kwargs['transforms'] = {'LegacyNorm': {}}
    if hasattr(args, 'ev_rate'):
        dataset_kwargs['ev_rate'] = args.ev_rate
    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type,
                                      normalize=args.normalize, num_workers=2)
    iter_dataloader = iter(data_loader)

    try:
        model.reset_states()
    except Exception as E:
        print(E)

    T = len(iter_dataloader)
    if args.pause_st is not None:
        T = args.max_length
    if args.max_length is not None:
        T = min(T, args.max_length)
    input_org_datas = [None] * T
    input_seqs = [None] * T
    predicts = [None] * T
    item_org = None
    crop = None
    for i in range(T):
        if args.pause_st is not None and i > args.pause_st and i <= args.pause_ed:
            item = copy.deepcopy(item_org)
            item['events'].fill_(0.0)
            if 'flow' in item:
                item['flow'].fill_(0.0)
            # item['events'] = add_noise_to_voxel(item['events'])
        else:
            item = next(iter_dataloader)
            item_org = copy.deepcopy(item)
        input_org_datas[i] = item
        # input_org_datas.append(item)
        if crop is None:
            height, width = item['frame'].shape[-2:]
            try:
                num_encoders = model.num_encoders
            except:
                num_encoders = 3
            crop = Croper(num_encoders)
            crop.update_params(width, height)

        voxel = item['events'].to(device)
        input_item = {'events': crop.pad(voxel)}
        if args.seq_model:
            input_seqs[i] = input_item
            # input_seqs.append(input_item)
        else:
            with torch.no_grad():
                output = model(input_item)
                predicts[i] = output
                # predicts.append(output)
    if args.seq_model:
        with torch.no_grad():
            if args.subseq_L is not None:
                predicts = []
                for subseqs in more_itertools.chunked(input_seqs, args.subseq_L):
                    predicts += model(subseqs)
            else:
                predicts = model(input_seqs)
            predicts = [{'image': img} for img in predicts]

    result = {}
    detail_res = collections.defaultdict(list)
    for loss in args.metrics:
        result[loss['name']] = 0

    for i in range(len(input_org_datas)):
        item = input_org_datas[i]
        voxel = item['events']
        image_gt = item['frame'].to(device)
        if args.loader_type == "SpadeH5":
            event_frame = torch.sum(voxel[0, 0], dim=0).squeeze()
            event_frame = quick_norm(event_frame).cpu().numpy()
        else:
            event_frame = torch.sum(voxel, dim=1).squeeze()
            event_frame = quick_norm(event_frame).cpu().numpy()

        output = predicts[i]

        image_float = crop.crop(output['image'])
        if 'SNN' in args.checkpoint_name:
            image_float = normalize_image(image_float)
        if args.eq:
            image_float = torch2cv2(image_float)
            image_float = cv2.equalizeHist(image_float)
            image_float = cv2torch(image_float)
            image_gt = torch2cv2(image_gt)
            image_gt = cv2.equalizeHist(image_gt)
            image_gt = cv2torch(image_gt)

        for l in args.metrics:
            name = l['name']
            l_func = l['func']
            loss_tmp = l_func(image_float, image_gt)
            result[name] += loss_tmp
            detail_res[name].append(float(loss_tmp.item() if type(loss_tmp) == torch.Tensor else loss_tmp))

        if args.showim or args.saveim:
            image_gt = image_gt.squeeze()
            image_float = image_float.squeeze()
            img_show = np.concatenate([event_frame, image_float.cpu().numpy(), image_gt.cpu().numpy()], axis=1)
            img_show = np.uint8(img_show*255)
            fname = 'frame_{:010d}.png'.format(i)

            if args.saveim:
                if not os.path.exists(args.output_folder):
                    os.makedirs(args.output_folder)
                cv2.imwrite(join(args.output_folder, fname), img_show)

            if args.showim:
                cv2.imshow('result', img_show)
                key = cv2.waitKey(args.wait_time) & 0xFF
                if key == 27:
                    sys.exit(0)

    for l in args.metrics:
        loss = result[l['name']]
        loss = loss.item() if type(loss) == torch.Tensor else loss
        result[l['name']] = loss / (len(input_org_datas))
    return result, detail_res


def haveKeys(data, keys):
    for key in keys:
        if key in data:
            return True
    return False


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args for eval_model.py")
    args.add_argument('--weights_dir', default=None, type=str)
    args.add_argument('--data_dir', default=None, type=str)
    args.add_argument('--st', default=0, type=int)
    args.add_argument('--ed', default=-1, type=int)
    args = args.parse_args()

    args.checkpoint_dir = "weights/"
    if args.weights_dir is not None:
        args.checkpoint_dir = args.weights_dir
    args.metrics = [{'func': perceptual_loss, 'name': 'p_loss'},
                    {'func': mse_loss, 'name': 'mse'},
                    {'func': structural_similarity, 'name': 'ssim'}]
    args.output_folder_root = "results/"
    args.saveim = False
    args.showim = True
    args.seq_model = False
    args.subseq_L = 1000  # 1000

    args.pause_st = None  # None, 200
    args.pause_ed = 1200  # None, 200
    # args.max_length = args.pause_ed + 200  # 3000
    args.max_length = 111200  # 3000
    args.loader_type = 'H5' # NPY, H5, ENPY
    args.datatype = 'org'  # ecoco, ergb, org, hsergb
    args.datasets = ['HQF', 'ECD', 'MVSEC', 'RAECD']  # ['HQF', 'IJRR', 'MVSEC', ] 'RAECD_GT' "ENFS" "CECOCO_ENPY"
    args.filter_hot_events = False
    args.normalize = False
    args.wait_time = 1
    args.eq = False

    # load all datas
    dataDir = "data/eval/"
    if args.data_dir is not None:
        dataDir = args.data_dir
    if args.loader_type == 'H5':
        subdir = 'h5'
    elif args.loader_type in ['NPY', 'ENPY']:
        subdir = 'npy'
    else:
        raise ValueError(f"Unknown loader type {args.loader_type}.")
    if args.datatype == "org":
        datafiles = "eval_data.txt"
    else:
        raise ValueError(f"Wrong datatype {args.datatype}")
    dataDir = os.path.join(dataDir, subdir)
    datafiles = os.path.join(dataDir, datafiles)
    with open(datafiles, 'r') as f:
        datafiles = f.read().split('\n')
        datafiles = list(filter(None, datafiles))
    haveKeys = functools.partial(haveKeys, keys=args.datasets)
    datafiles = list(filter(haveKeys, datafiles))

    def read_cps():
        checkpoint_files = glob(os.path.join(args.checkpoint_dir, "*.pth"))
        def parse_num(x):
            if 'epoch_' in x:
                x = x.split('epoch_')[-1]
                x = int(re.search(r'\d+', x)[0])
            return x
        checkpoint_files.sort(key=parse_num)
        if args.st > 0 or args.ed > -1:
            st = args.st if args.st > 0 else 0
            ed = args.ed if args.ed > -1 else len(checkpoint_files)
            checkpoint_files = checkpoint_files[st:ed]
        return checkpoint_files

    checkpoint_files = read_cps()
    for checkpoint_file in checkpoint_files:
        argsn = copy.deepcopy(args)
        argsn.checkpoint_path = checkpoint_file
        eval_model_alldata(datafiles, checkpoint_file, argsn)
