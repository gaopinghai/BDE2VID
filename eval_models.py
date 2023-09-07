import functools
import gc
import cv2
import os, sys
from os.path import join
import numpy as np
import argparse
from glob import glob
import copy
import collections
import json
from tqdm import tqdm

import torch
from mmengine.config import Config
from mmengine.registry import MODELS

from model.spade_e2vid.spade_e2v import SPADEE2VID
from model.e2vid_seq.e2vid_seq import E2VIDSeq
from model.e2vid.model import FlowNet, E2VIDRecurrent
from model.EVSNN.model.snn_network import EVSNN_LIF_final, PAEVSNN_LIF_AMPLIF_final
from model.eitr.eitr import EITR
from model.EVSNN.rec_snn_forward import RecSNN
from model.EVSNN.utils.util import normalize_image
from evaluate.metrics import mse_loss, structural_similarity, perceptual_loss
from evaluate.generate_table import generate_table
from data_loader.h5_dataset import InferenceDataLoader
from utils_func.inference_utils import Croper, torch2cv2, cv2torch
from utils_func.utils import quick_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpointfile, args=None, **kwargs):
    """用于加载各种模型的总函数"""
    checkpoint = torch.load(checkpointfile)
    model = None
    if 'state_dict' in checkpoint:
        if 'meta' in checkpoint:
            model_cfg = checkpoint['meta']['cfg']
            model_cfg = Config.fromstring(model_cfg, '.py')
            model_cfg = model_cfg.model
            if 'type' in model_cfg:
                # load E2VIDSeq
                model = MODELS.build(model_cfg)
                args.seq_model = True
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
                    else:
                        model_args = model_args['eitr_kwargs']
            elif 'model' in checkpoint:
                model_args = checkpoint['model']
                try:
                    args.normalize = True
                except:
                    pass
            model = eval(arch)(model_args)
        model.load_state_dict(checkpoint['state_dict'], True)
    else:
        if "SPADE" in checkpointfile:
            model = SPADEE2VID()
            model.load_state_dict(checkpoint)
            args.normalize = True
            args.loader_type = "SpadeH5"
            args.ev_rate = 0.35
        elif "SNN" in checkpointfile:
            model_name = "PAEVSNN_LIF_AMPLIF_final" if "PAEVSNN" in checkpoint_file \
                else "EVSNN_LIF_final"
            model = RecSNN(model_name, checkpoint_file)
    return model


def eval_model_alldata(datafiles, checkpoint_file, args):
    args.checkpoint_name = os.path.split(args.checkpoint_path)[-1].split('.')[0]
    result_file = f"{args.checkpoint_name}.txt"
    checkpoint_dir = f"{args.checkpoint_name}"
    result_file = os.path.join(args.checkpoint_dir, result_file)
    if os.path.exists(result_file):
        print(f"skiping {checkpoint_file}")
        return
    else:
        # 在此处生成结果文件防止其他程序重复处理该模型权重
        fp = open(result_file, 'w')

    print('Loading checkpoint: {} ...'.format(checkpoint_file))
    model = load_model(checkpoint_file, args)
    model.eval()
    model.to(device)
    results = collections.defaultdict(lambda: collections.defaultdict(dict))
    detail_results = collections.defaultdict(lambda: collections.defaultdict(dict))
    for datafile in datafiles:
        args.events_file_path = os.path.join(dataDir, datafile)
        datasetName, filename = datafile.split(os.sep)
        filename = filename.split('.')[0]
        args.output_folder = os.path.join(args.output_folder_root, checkpoint_dir, datasetName, filename)
        result, detail_res = eval_model(args, model)
        results[datasetName][filename] = result
        detail_results[datasetName][filename] = detail_res

    json.dump(results, fp)
    fp.close()
    print(f"results writed to {result_file}")
    generate_table([result_file], result_file.replace('.txt', '_table.txt'))
    # detail_resfile = result_file.replace('.txt', '_detail.txt')
    # fp = open(detail_resfile, 'w')
    # json.dump(detail_results, fp)
    # fp.close()


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
                                      normalize=args.normalize, num_workers=1)
    iter_dataloader = iter(data_loader)

    #################################################
    ##########     进行预测
    try:
        model.reset_states()
    except Exception as E:
        print(E)

    T = len(iter_dataloader)
    input_org_datas = [None]*T
    input_seqs = [None]*T
    predicts = [None]*T
    item_org = None
    crop = None
    for i in tqdm(range(T)):
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
    del iter_dataloader, data_loader
    gc.collect()
    if args.seq_model:
        with torch.no_grad():
            predicts = model(input_seqs)
            predicts = [{'image': img} for img in predicts]
    ########     预测结束
    ###################################################################

    ###################################################################
    ########    计算loss
    result = {}
    detail_res = collections.defaultdict(list)
    for loss in args.metrics:
        result[loss['name']] = 0

    for i in range(len(input_org_datas)):
        item = input_org_datas[i]
        voxel = item['events'].to(device)
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
        result[l['name']] = loss / (i + 1.0)
    return result, detail_res


def haveKeys(data, keys):
    for key in keys:
        if key in data:
            return True
    return False


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="args for eval_model.py")
    args.add_argument('-d', '--dir', default=None, type=str)
    args = args.parse_args()

    args.checkpoint_dir = "weights"
    if args.dir is not None:
        args.checkpoint_dir = args.dir
    args.metrics = [{'func': perceptual_loss, 'name': 'p_loss'},
                    {'func': mse_loss, 'name': 'mse'},
                    {'func': structural_similarity, 'name': 'ssim'}]
    args.output_folder_root = "results/"
    args.saveim = False
    args.showim = False
    args.seq_model = False

    args.filter_hot_events = False
    args.normalize = False
    args.wait_time = 1
    args.eq = False

    # load all datas
    args.loader_type = "H5"
    dataDir = "/home/gph/JZSSD/Dataset/ecoco_depthmaps/eval/h5"
    datafiles = os.path.join(dataDir, "eval_data.txt")
    with open(datafiles, 'r') as f:
        datafiles = f.read().split('\n')
        datafiles = list(filter(None, datafiles))

    checkpoint_files = glob(os.path.join(args.checkpoint_dir, "*.pth*"))
    checkpoint_files.sort()
    for checkpoint_file in checkpoint_files:
        argsn = copy.deepcopy(args)
        argsn.checkpoint_path = checkpoint_file
        eval_model_alldata(datafiles, checkpoint_file, argsn)
