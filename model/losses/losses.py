import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error

from mmengine.registry import MODELS
LOSSES = MODELS

from LPIPS.models import dist_model as dm


@LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, x, y):
        return self.loss(x, y)


@LOSSES.register_module()
class PerceptualLoss(nn.Module):
    def __init__(self, net='alex', model='net-lin', use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.model = dm.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu)

    def forward(self, pred, target, normalize=True):
        # pred and target are N x C x H x W in the range [0,1]
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        if pred.shape[1] == 1:
            pred_c3 = torch.cat([pred, pred, pred], dim=1)
            target_c3 = torch.cat([target, target, target], dim=1)
        else:
            pred_c3 = pred
            target_c3 = target

        dist = self.model.forward_pair(pred_c3, target_c3)
        return dist.mean()


@LOSSES.register_module()
class TemporalConsistencyLoss(nn.Module):
    def __init__(self, L0=2, weight=1):
        super(TemporalConsistencyLoss, self).__init__()
        self.L0 = L0
        self.weight = weight

    def forward(self, image0, image1, processed0, processed1, flow01):
        return temporal_consistency_loss(image0, image1, processed0, processed1, flow01)*self.weight


def temporal_consistency_loss(image0, image1, processed0, processed1, flow01,
                              alpha=50.0, output_images=False):
    """ Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video
        Temporal Consistency', Lai et al., ECCV'18.

        The temporal loss is the warping error between two processed frames
        (image reconstructions in E2VID),
        after the images have been aligned using the flow `flow01`.
        The input (ground truth) images `image0` and `image1` are used to estimate a
        visibility mask.

        :param image0: [N x C x H x W] input image 0
        :param image1: [N x C x H x W] input image 1
        :param processed0: [N x C x H x W] processed image (reconstruction) 0
        :param processed1: [N x C x H x W] processed image (reconstruction) 1
        :param flow01: [N x 2 x H x W] displacement map from image1 to image0
        :param alpha: used for computation of the visibility mask (default: 50.0)
    """
    t_width, t_height = image0.shape[3], image0.shape[2]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    xx, yy = xx.to(image0.device), yy.to(image0.device)
    xx.transpose_(0, 1)
    yy.transpose_(0, 1)
    xx, yy = xx.float(), yy.float()

    flow01_x = flow01[:, 0, :, :]  # N x H x W
    flow01_y = flow01[:, 1, :, :]  # N x H x W

    warping_grid_x = xx + flow01_x  # N x H x W
    warping_grid_y = yy + flow01_y  # N x H x W

    # normalize warping grid to [-1,1]
    warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
    warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

    warping_grid = torch.stack(
        [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

    image0_warped_to1 = F.grid_sample(image0, warping_grid)
    visibility_mask = torch.exp(-alpha * (image1 - image0_warped_to1) ** 2)
    processed0_warped_to1 = F.grid_sample(processed0, warping_grid)

    tc_map = visibility_mask * torch.abs(processed1 - processed0_warped_to1) \
             / (torch.abs(processed1) + torch.abs(processed0_warped_to1) + 1e-5)

    tc_loss = tc_map.mean()

    if output_images:
        additional_output = {'image0_warped_to1': image0_warped_to1,
                             'processed0_warped_to1': processed0_warped_to1,
                             'visibility_mask': visibility_mask,
                             'error_map': tc_map}
        return tc_loss, additional_output

    else:
        return tc_loss