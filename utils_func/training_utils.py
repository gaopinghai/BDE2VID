import numpy as np
import cv2
import torch


def flow2rgb(disp_x, disp_y, max_magnitude=None):
    """
    Convert an optic flow tensor to an RGB color map for visualization
    Code adapted from: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py#L339

    :param disp_x: a [H x W] NumPy array containing the X displacement
    :param disp_x: a [H x W] NumPy array containing the Y displacement
    :returns bgr: a [H x W x 3] NumPy array containing a color-coded representation of the flow
    """
    assert(disp_x.shape == disp_y.shape)
    H, W = disp_x.shape

    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))

    flow_x = (X - disp_x) * float(W) / 2
    flow_y = (Y - disp_y) * float(H) / 2
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    if max_magnitude is None:
        v = np.zeros(magnitude.shape, dtype=np.uint8)
        cv2.normalize(src=magnitude, dst=v, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        v = 255.0 * magnitude / max_magnitude
        v = v.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = 0.5 * angle * 180 / np.pi
    hsv[..., 2] = v
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def flow2rgb_torch(disp_x, disp_y, max_magnitude=None):
    device = disp_x.device
    rgb = flow2rgb(disp_x.cpu().numpy(), disp_y.cpu().numpy(), max_magnitude)
    rgb = rgb.astype(float) / 255
    return torch.tensor(rgb).to(device)  # 3 x H x W
