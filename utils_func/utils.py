import torch
from pathlib import Path
import json
from collections import OrderedDict


class RobustNorm(object):
    """
    Robustly normalize tensor
    """

    def __init__(self, low_perc=0, top_perc=95):
        self.top_perc = top_perc
        self.low_perc = low_perc

    @staticmethod
    def percentile(t, q):
        """
        Return the ``q``-th percentile of the flattened input tensor's data.

        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.

        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        try:
            result = t.view(-1).kthvalue(k).values.item()
        except RuntimeError:
            result = t.reshape(-1).kthvalue(k).values.item()
        return result

    def __call__(self, x, is_flow=False):
        """
        """
        t_max = self.percentile(x, self.top_perc)
        t_min = self.percentile(x, self.low_perc)
        # print("t_max={}, t_min={}".format(t_max, t_min))
        if t_max == 0 and t_min == 0:
            return x
        eps = 1e-6
        normed = torch.clamp(x, min=t_min, max=t_max)
        normed = (normed - torch.min(normed)) / (torch.max(normed) + eps)
        return normed
robust_1_99 = RobustNorm(1, 99)


def abs_norm(data):
    data -= data.min()
    if data.max() != 0:
        data /= data.max()
    return data


def quick_norm(img):
    return (img - torch.min(img))/(torch.max(img) - torch.min(img) + 1e-6)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
