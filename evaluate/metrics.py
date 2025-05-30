from os.path import basename, dirname
from tqdm import tqdm
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import numpy as np
from LPIPS.models import dist_model as dm

import torch.nn.functional as F


@METRICS.register_module()
class Mse(BaseMetric):
    default_prefix = "MSE"
    def __init__(self, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device,
                         prefix=prefix)

    def process(self, data_batch, predictions):
        base_folder = data_batch.get('base_folder', None)[0]
        if base_folder is not None:
            seq_name, dataset_name = basename(base_folder), basename(dirname(base_folder))
        else:
            seq_name, dataset_name = 'unknown', 'unknown'
        # print(f"calcing {self.default_prefix} for {dataset_name}/{seq_name}")

        preds, gts = predictions
        L = len(preds)
        loss = F.mse_loss(torch.cat(preds), torch.cat(gts))
        self.results.append({self.default_prefix: loss.item(), 'L': L,
                             "seq_name": seq_name, 'dataset': dataset_name})

    def compute_metrics(self, results):
        mse_all = sum([res[self.default_prefix]*res['L'] for res in results])
        L = sum([res['L'] for res in results])
        loss = mse_all / L
        return {self.default_prefix: loss}


def mse_loss(y_input, y_target):
    return F.mse_loss(y_input, y_target)


def structural_similarity(y_input, y_target):
    y_input = y_input.cpu().numpy()
    y_target = y_target.cpu().numpy()
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    # N x C x H x W -> N x W x H x C -> N x H x W x C
    y_input = np.swapaxes(y_input, 1, 3)
    y_input = np.swapaxes(y_input, 1, 2)
    y_target = np.swapaxes(y_target, 1, 3)
    y_target = np.swapaxes(y_target, 1, 2)
    sum_structural_similarity_over_batch = 0.
    for i in range(N):
        if C == 3:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, :], y_target[i, :, :, :], multichannel=True)
        else:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, 0], y_target[i, :, :, 0])

    return sum_structural_similarity_over_batch / float(N)


""" Perceptual distance """
class PerceptualLoss(torch.nn.Module):
    # VGG using our perceptually-learned weights (LPIPS metric)
    def __init__(self, model='net-lin', net='alex', use_gpu=True):
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss..')
        self.model = dm.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu)
        print('Done')

    def forward(self, pred, target, normalize=True):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        if pred.shape[1] == 1:
            pred_c3 = torch.cat([pred, pred, pred], dim=1)
            target_c3 = torch.cat([target, target, target], dim=1)
        else:
            pred_c3 = pred
            target_c3 = target

        dist = self.model.forward_pair(target_c3, pred_c3)

        return dist.mean()


perceptual_loss = PerceptualLoss()
