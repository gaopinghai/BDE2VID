# -*- coding: utf-8 -*-
'''
This code is provided for internal research and development purposes by Huawei solely,
in accordance with the terms and conditions of the research collaboration agreement of May 7, 2020.
Any further use for commercial purposes is subject to a written agreement.
'''
"""
Dataset classes
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from torch.utils.data import Dataset
from .event_dataset import VoxelGridDataset
from skimage import io
import cv2
from os.path import join
import numpy as np
from data_loader.dataloader_utils import first_element_greater_than, last_element_less_than
from utils_func.utils import abs_norm, RobustNorm
import random
import glob
import torch
from math import fabs

from events_contrast_maximization.utils.event_utils import events_to_voxel_torch,\
    events_to_neg_pos_voxel_torch


robustNorm = RobustNorm(low_perc=0, top_perc=99)


class SequenceSynchronizedFramesEventsDataset(Dataset):
    """Load sequences of time-synchronized {event tensors + frames} from a folder."""

    def __init__(self, base_folder, event_folder, frame_folder='frames', flow_folder='flow',
                 start_time=0.0, stop_time=0.0, num_bins=5, subdset=None, min_event_rate=1,
                 sequence_length=2, allow_shoter=False, transform=None,
                 proba_pause_when_running=0.0, proba_pause_when_paused=0.0,
                 step_size=20,
                 normalize=True, abs_norm=False,
                 mask_args={}):
        self.abs_norm = abs_norm
        self.mask_size = mask_args.get("mask_size", None)
        if self.mask_size is not None:
            assert self.mask_size[0] < self.mask_size[1]
        self.noise_std = mask_args.get("noise_std", 0)
        self.noise_fraction = mask_args.get("noise_fraction", 0)
        self.hot_pixel_std = mask_args.get("hot_pixel_std", 0)
        self.max_hot_pixel_fraction = mask_args.get("max_hot_pixel_fraction", 0)
        self.proba_mask_when_running = mask_args.get("proba_mask_when_running", 0)
        self.proba_mask_when_masked = mask_args.get("proba_mask_when_masked", 0)
        self.proba_noise_when_pause = mask_args.get("proba_noise_when_pause", 1)
        assert(step_size > 0)
        self.L = sequence_length
        if subdset == "SynchronizedNPYDataset":
            self.dataset = SynchronizedNPYDataset(base_folder, event_folder, frame_folder, flow_folder,
                                                  start_time, stop_time, transform, min_event_rate=min_event_rate,
                                                  num_bins=num_bins, normalize=normalize)
        else:
            self.dataset = SynchronizedFramesEventsDataset(base_folder, event_folder, frame_folder, flow_folder,
                                                           start_time, stop_time,
                                                           transform, normalize=normalize)
        if self.L < 0:
            self.L = self.dataset.length
        if allow_shoter:
            self.L = min(self.L, self.dataset.length)
        assert (self.L > 0)
        self.event_dataset = self.dataset.event_dataset
        self.step_size = step_size
        if self.L > self.dataset.length:
            self.length = 0
        else:
            self.length = (self.dataset.length - self.L) // self.step_size + 1

        self.proba_pause_when_running = proba_pause_when_running
        self.proba_pause_when_paused = proba_pause_when_paused

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """ Returns a list containing synchronized events <-> frame pairs
            [e_{i-L} <-> I_{i-L},
             e_{i-L+1} <-> I_{i-L+1},
            ...,
            e_{i-1} <-> I_{i-1},
            e_i <-> I_i]
        """
        assert(i >= 0)
        assert(i < self.length)

        # generate a random seed here, that we will pass to the transform function
        # of each item, to make sure all the items in the sequence are transformed
        # in the same way
        seed = random.randint(0, 2**32)

        # data augmentation: add random, virtual "pauses",
        # i.e. zero out random event tensors and repeat the last frame
        sequence = []

        # add the first element (i.e. do not start with a pause)
        k = 0
        j = i * self.step_size

        # with ThreadPoolExecutor() as excutor:
        #     results = [excutor.submit(partial(self.dataset.__getitem__, seed=seed), ind) for ind in range(j, j+self.L)]
        #     sequence_org = [res.result() for res in results]

        item = self.dataset.__getitem__(j, seed)
        # item = sequence_org[j]
        sequence.append(item)

        noise_when_pause = np.random.rand() < self.proba_noise_when_pause
        mask = None
        masked = False
        paused = False
        for n in range(1, self.L):
            # decide whether we should make a "pause" at this step
            # the probability of "pause" is conditioned on the previous state (to encourage long sequences)
            u = np.random.rand()
            if paused:
                probability_pause = self.proba_pause_when_paused
            else:
                probability_pause = self.proba_pause_when_running
            paused = (u < probability_pause)

            if paused:
                # add a tensor filled with zeros, paired with the last frame
                # do not increase the counter
                item = self.dataset.__getitem__(j + k, seed)
                # item = sequence_org[j+k]
                item['events'].fill_(0.0)
                # item['events'] = self.add_noise_to_voxel(item['events'])
                if 'flow' in item:
                    item['flow'].fill_(0.0)
            else:
                # normal case: append the next item to the list
                k += 1
                item = self.dataset.__getitem__(j + k, seed)
                # item = sequence_org[j+k]
                paused = False

            # u = np.random.rand()
            # if masked:
            #     proba_mask = self.proba_mask_when_masked
            # else:
            #     proba_mask = self.proba_mask_when_running
            # masked = (u < proba_mask)
            # if masked:
            #     if not paused or (paused and noise_when_pause):
            item['events'] = self.add_noise_to_voxel(item['events'])
            # item['events'] = abs_norm(item['events'])
            sequence.append(item)

        # add hot pixels
        if self.max_hot_pixel_fraction > 0:
            add_hot_pixels_to_sequence_(sequence, self.hot_pixel_std, self.max_hot_pixel_fraction)

        if self.abs_norm:
            abs_norm_sequence_(sequence)

        return {'inputs': sequence, 'base_folder': self.dataset.base_folder}

    def mask_evemts(self, events, mask):
        if mask is None:
            H, W = events.shape[-2:]
            pk = lambda: np.random.rand()*(self.mask_size[1] - self.mask_size[0]) + self.mask_size[0]
            h, w = int(H * pk()), int(W * pk())
            x = np.random.randint(0, H-h)
            y = np.random.randint(0, W-w)
            mask = torch.zeros((H, W))
            mask[y:y+h, x:x+w] = 1.0
        noise = self.noise_std * torch.randn_like(mask)
        events = events * (1 - mask) + noise * mask
        return events, mask

    def add_noise_to_voxel(self, voxel):
        noise_std = self.noise_std
        noise_fraction = self.noise_fraction
        noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
        if noise_fraction < 1.0:
            mask = torch.rand_like(voxel) >= noise_fraction
            noise.masked_fill_(mask, 0)
        return voxel + noise


def abs_norm_sequence_(sequence):
    for item in sequence:
        item['events'] = robustNorm(item['events'])


def add_hot_pixels_to_sequence_(sequence, hot_pixel_std=1.0, max_hot_pixel_fraction=0.001):
    hot_pixel_fraction = random.uniform(0, max_hot_pixel_fraction)
    voxel = sequence[0]['events']
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    val = torch.randn(num_hot_pixels, dtype=voxel.dtype, device=voxel.device)
    val *= hot_pixel_std
    # TODO multiprocessing
    for item in sequence:
        for i in range(num_hot_pixels):
            item['events'][..., :, y[i], x[i]] += val[i]


class SynchronizedNPYDataset(Dataset):
    """Loads time-synchronized event tensors and frames from a folder.

    This Dataset class iterates through all the event tensors and returns, for each tensor,
    a dictionary of the form:

        {'frame': frame, 'events': events, 'flow': disp_01}

    where:

    * frame is a H x W tensor containing the first frame whose timestamp >= event tensor
    * events is a C x H x W tensor containing the event data
    * flow is a 2 x H x W tensor containing the flow (displacement) from the current frame to the last frame

    This loader assumes that the event folder contains original events.
    """

    def __init__(self, base_folder, event_folder, frame_folder='frames', flow_folder='flow',
                 start_time=0.0, stop_time=0.0, transform=None, combined_voxel_channels=True,
                 num_bins=5, min_event_rate=1, normalize=False):
        self.normalize = normalize
        self.base_folder = base_folder
        self.num_bins = num_bins
        self.min_event_rate = min_event_rate
        self.sensor_resolution = None
        self.event_dataset = None
        self.combined_voxel_channels = combined_voxel_channels
        self.frame_folder = join(self.base_folder, frame_folder if frame_folder is not None else 'frames')
        self.event_folder = join(self.base_folder, event_folder)
        if flow_folder is not None:
            self.flow_folder = join(self.base_folder, flow_folder)
        else:
            self.flow_folder = None
        self.transform = transform

        self.frames = None
        self.flows = None

        # Load the stamp files
        self.stamps = np.loadtxt(join(self.frame_folder, 'frame_ts.txt'))

        self.length = len(self.stamps)

        # Check that the frame timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)
               ), "frame timestamps are not unique and monotonically increasing"

    def __len__(self):
        return self.length

    def get_empty_voxel_grid(self, combined_voxel_channels=True):
        """Return an empty voxel grid filled with zeros"""
        if combined_voxel_channels:
            size = (self.num_bins, *self.sensor_resolution)
        else:
            size = (2*self.num_bins, *self.sensor_resolution)
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, combined_voxel_channels=True):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param combined_voxel_channels: if True, create voxel grid merging positive and
            negative events (resulting in NUM_BINS x H x W tensor). Otherwise, create
            voxel grid for positive and negative events separately
            (resulting in 2*NUM_BINS x H x W tensor)
        """
        if combined_voxel_channels:
            # generate voxel grid which has size self.num_bins x H x W
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        else:
            # generate voxel grid which has size 2*self.num_bins x H x W
            voxel_grid = events_to_neg_pos_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                                       sensor_size=self.sensor_resolution)
            voxel_grid = torch.cat([voxel_grid[0], voxel_grid[1]], 0)

        # voxel_grid = voxel_grid*self.hot_events_mask

        return voxel_grid

    def __getitem__(self, i, seed=None):
        assert(i >= 0)
        assert(i < self.length)
        if self.frames is None:
            self.frames = np.load(join(self.frame_folder, "frames.npy"), mmap_mode='r')
            self.sensor_resolution = self.frames.shape[-2:]
            self.min_event_num = int(self.min_event_rate * np.prod(self.sensor_resolution))
            self.xs = np.load(join(self.event_folder, "xs.npy"), mmap_mode='r')
            self.ys = np.load(join(self.event_folder, "ys.npy"), mmap_mode='r')
            self.ts = np.load(join(self.event_folder, "ts.npy"), mmap_mode='r')
            self.ps = np.load(join(self.event_folder, "ps.npy"), mmap_mode='r')
            self.event_idxes = np.loadtxt(join(self.frame_folder, "frame_event_idxes.txt"), dtype=np.int_)
        if self.flow_folder is not None and self.flows is None:
            self.flows = np.load(join(self.flow_folder, "flows.npy"), mmap_mode='r')
        frames, flows = self.frames, self.flows

        idx0, idx1 = self.event_idxes[i]
        xs, ys, ts, ps = self.xs[idx0:idx1], self.ys[idx0:idx1], self.ts[idx0:idx1], self.ps[idx0:idx1]
        ps = ps * 2 - 1
        if self.min_event_num < len(xs):
            events_num = np.random.randint(self.min_event_num, len(xs)+1)
            idxes = np.arange(0, len(xs))
            idxes = np.random.choice(idxes, events_num, replace=False)
            idxes.sort()
            xs, ys, ts, ps = xs[idxes], ys[idxes], ts[idxes], ps[idxes]
        try:
            ts_0, ts_k = ts[0], ts[-1]
        except:
            ts_0, ts_k = 0, 0
        if len(xs) < 3:
            voxel = self.get_empty_voxel_grid(self.combined_voxel_channels)
        else:
            xs = torch.from_numpy(xs.astype(np.float32))
            ys = torch.from_numpy(ys.astype(np.float32))
            ts = torch.from_numpy((ts - ts_0).astype(np.float32))
            ps = torch.from_numpy(ps.astype(np.float32))
            voxel = self.get_voxel_grid(xs, ys, ts, ps, combined_voxel_channels=self.combined_voxel_channels)
        if self.normalize:
            Warning(f"Haven't realize normalize.")

        events = voxel

        frame_timestamp = self.stamps[i]
        frame = frames[i]
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.
        if len(frame.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame = np.expand_dims(frame, -1)
        frame = np.moveaxis(frame, -1, 0)  # H x W x C -> C x H x W
        frame = torch.from_numpy(frame)

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2**32)
        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)
            random.seed(seed)
            events = self.transform(events)
        events = {'events': events}
        # Get optic flow tensor and apply the same transformation as the others
        if self.flow_folder is not None:
            try:
                # flow = np.load(join(self.flow_folder, 'disp01_{:010d}.npy'.format(i + 1))).astype(np.float32)
                flow = flows[i].astype(np.float32)
                flow = torch.from_numpy(flow)  # [2 x H x W]
                if self.transform:
                    random.seed(seed)
                    flow = self.transform(flow, is_flow=True)
            except FileNotFoundError:
                flow = None
        else:
            flow = None

        # Merge the 'frame' dictionary with the 'events' one
        if flow is not None:
            item = {'frame': frame,
                    'flow': flow,
                    'timestamp': frame_timestamp,
                    **events}
        else:
            item = {'frame': frame,
                    'timestamp': frame_timestamp,
                    **events}

        return item


class SynchronizedFramesEventsDataset(Dataset):
    """Loads time-synchronized event tensors and frames from a folder.

    This Dataset class iterates through all the event tensors and returns, for each tensor,
    a dictionary of the form:

        {'frame': frame, 'events': events, 'flow': disp_01}

    where:

    * frame is a H x W tensor containing the first frame whose timestamp >= event tensor
    * events is a C x H x W tensor containing the event data
    * flow is a 2 x H x W tensor containing the flow (displacement) from the current frame to the last frame

    This loader assumes that each event tensor can be uniquely associated with a frame.
    For each event tensor with timestamp e_t, the corresponding frame is the first frame whose timestamp f_t >= e_t

    """

    def __init__(self, base_folder, event_folder, frame_folder='frames', flow_folder='flow',
                 start_time=0.0, stop_time=0.0,
                 transform=None,
                 normalize=True):

        self.base_folder = base_folder
        self.frame_folder = join(self.base_folder, frame_folder if frame_folder is not None else 'frames')
        if flow_folder is not None:
            self.flow_folder = join(self.base_folder, flow_folder)
        else:
            self.flow_folder = None
        self.transform = transform
        self.event_dataset = VoxelGridDataset(base_folder, event_folder,
                                              start_time, stop_time,
                                              transform=self.transform,
                                              normalize=normalize)

        self.frames = None
        self.flows = None

        # Load the stamp files
        self.stamps = np.loadtxt(
            join(self.frame_folder, 'timestamps.txt'))[:, 1]

        # shift the frame timestamps by the same amount as the event timestamps
        self.stamps -= self.event_dataset.initial_stamp

        self.length = len(self.event_dataset)

        # Check that the frame timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)
               ), "frame timestamps are not unique and monotonically increasing"

        # Check that the latest frame in the dataset has a timestamp >= the latest event frame
        assert(self.stamps[-1] >= self.event_dataset.get_last_stamp())

    def __len__(self):
        return self.length

    def __getitem__(self, i, seed=None):
        assert(i >= 0)
        assert(i < self.length)
        if self.frames is None:
            self.frames = np.load(join(self.frame_folder, "frames.npy"), mmap_mode='r')
        if self.flow_folder is not None and self.flows is None:
            self.flows = np.load(join(self.flow_folder, "flows.npy"), mmap_mode='r')
        frames, flows = self.frames, self.flows

        event_timestamp = self.event_dataset.get_stamp_at(i)

        # Find the index of the first frame whose timestamp is >= event timestamp
        (frame_idx, frame_timestamp) = first_element_greater_than(
            self.stamps, event_timestamp)
        assert(frame_idx >= 0)
        assert(frame_idx < len(self.stamps))
        # assert(frame_timestamp >= event_timestamp)
        assert(frame_timestamp >= event_timestamp)
        # tol = 0.01
        # if fabs(frame_timestamp - event_timestamp) > tol:
        #     print(
        #         'Warning: frame_timestamp and event_timestamp differ by more than tol ({} s)'.format(tol))
        #     print('frame_timestamp = {}, event_timestamp = {}'.format(
        #         frame_timestamp, event_timestamp))

        # frame = io.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame_idx)),
        #                   as_gray=False).astype(np.float32) / 255.
        # frame = cv2.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame_idx)),
        #                    cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        frame = frames[frame_idx]
        if frame.dtype == np.uint8:
            frame = frames[frame_idx].astype(np.float32) / 255.

        if len(frame.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame = np.expand_dims(frame, -1)

        frame = np.moveaxis(frame, -1, 0)  # H x W x C -> C x H x W
        frame = torch.from_numpy(frame)

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2**32)

        if self.transform:
            random.seed(seed)
            frame = self.transform(frame)

        # Get the event tensor from the event dataset loader
        # Note that we pass the transform seed to ensure the same transform is used
        events = self.event_dataset.__getitem__(i, seed)

        # Get optic flow tensor and apply the same transformation as the others
        if self.flow_folder is not None:
            try:
                # flow = np.load(join(self.flow_folder, 'disp01_{:010d}.npy'.format(i + 1))).astype(np.float32)
                flow = flows[i].astype(np.float32)
                flow = torch.from_numpy(flow)  # [2 x H x W]
                if self.transform:
                    random.seed(seed)
                    flow = self.transform(flow, is_flow=True)
            except FileNotFoundError:
                flow = None
        else:
            flow = None

        # Merge the 'frame' dictionary with the 'events' one
        if flow is not None:
            item = {'frame': frame,
                    'flow': flow,
                    'timestamp': frame_timestamp,
                    **events}
        else:
            item = {'frame': frame,
                    'timestamp': frame_timestamp,
                    **events}

        return item


class EventsBetweenFramesDataset(Dataset):
    """Loads time-synchronized event tensors and frame-pairs from a folder.

    This Dataset class iterates through all the event tensors and returns, for each tensor,
    a dictionary of the form:

        {'frames': [frame0, frame1], 'events': events}

    where:

    * frames is a tuple containing two H x W tensor containing the start/end frames
    * events is a C x H x W tensor containing the events in that were triggered in between the frames

    This loader assumes that each event tensor can be uniquely associated with a frame pair.
    For each event tensor with timestamp e_t, the corresponding frame pair is [frame_idx-1, frame_idx], where
    frame_idx is the index of the first frame whose timestamp f_t >= e_t

    """

    def __init__(self, base_folder, event_folder, frame_folder='frames',
                 start_time=0.0, stop_time=0.0, transform=None, normalize=True):
        self.base_folder = base_folder
        self.frame_folder = join(self.base_folder, frame_folder if frame_folder is not None else 'frames')
        self.transform = transform

        self.event_dataset = VoxelGridDataset(base_folder, event_folder,
                                              start_time, stop_time,
                                              transform=self.transform,
                                              normalize=normalize)

        # Load the frame stamps file
        self.stamps = np.loadtxt(
            join(self.frame_folder, 'timestamps.txt'))[:, 1]

        # Shift the frame timestamps by the same amount as the event timestamps
        self.stamps -= self.event_dataset.initial_stamp

        self.length = len(self.event_dataset)

        # Check that the frame timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)
               ), "frame timestamps are not unique and monotonically increasing"

        # Load the event boundaries stamps file
        # (it is a file containing the index of the first/last event for every event tensor)
        self.boundary_stamps = np.loadtxt(
            join(self.event_dataset.event_folder, 'boundary_timestamps.txt'))[:, 1:]

        # Shift the boundary timestamps by the same amount as the event timestamps
        self.boundary_stamps[:, 0] -= self.event_dataset.initial_stamp
        self.boundary_stamps[:, 1] -= self.event_dataset.initial_stamp

        # Check the the first event timestamp >= the first frame in the dataset
        assert(
            self.stamps[0] <= self.boundary_stamps[self.event_dataset.first_valid_idx, 0])

        # Check that the latest frame in the dataset has a timestamp >= the latest event frame
        assert(
            self.stamps[-1] >= self.boundary_stamps[self.event_dataset.last_valid_idx, 1])

    def __len__(self):
        return self.length

    def __getitem__(self, i, seed=None):
        assert(i >= 0)
        assert(i < self.length)

        # stamp of the first event in the tensor
        et0 = self.boundary_stamps[self.event_dataset.get_index_at(i), 0]
        # stamp of the last event in the tensor
        et1 = self.boundary_stamps[self.event_dataset.get_index_at(i), 1]

        # print('i = ', i)
        # print('et0, et1 = ', et0 + self.event_dataset.initial_stamp,
        #       et1 + self.event_dataset.initial_stamp)

        # Find the index of the last frame whose timestamp is <= et0
        (frame0_idx, frame0_timestamp) = last_element_less_than(
            self.stamps, et0)
        assert(frame0_idx >= 0)
        assert(frame0_idx < len(self.stamps))
        assert(frame0_timestamp <= et0)

        tol = 0.01
        if fabs(frame0_timestamp - et0) > tol:
            print(
                'Warning: frame0_timestamp and et0 differ by more than tol ({} s)'.format(tol))
            print('frame0_timestamp = {}, et0 = {}'.format(
                frame0_timestamp, et0))

        # frame0 = io.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame0_idx)),
        #                    as_gray=False).astype(np.float32) / 255.
        frame0 = cv2.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame0_idx)),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.

        if len(frame0.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame0 = np.expand_dims(frame0, -1)

        frame0 = np.moveaxis(frame0, -1, 0)  # H x W x C -> C x H x W
        frame0 = torch.from_numpy(frame0)

        # Find the index of the first frame whose timestamp is >= et1
        (frame1_idx, frame1_timestamp) = first_element_greater_than(
            self.stamps, et1)
        assert(frame1_idx >= 0)
        assert(frame1_idx < len(self.stamps))
        assert(frame1_timestamp >= et1)

        if fabs(frame1_timestamp - et1) > tol:
            print(
                'Warning: frame1_timestamp and et1 differ by more than tol ({} s)'.format(tol))
            print('frame1_timestamp = {}, et1 = {}'.format(
                frame1_timestamp, et1))

        # frame1 = io.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame1_idx)),
        #                    as_gray=False).astype(np.float32) / 255.
        frame1 = cv2.imread(join(self.frame_folder, 'frame_{:010d}.png'.format(frame1_idx)),
                            cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        frame1 = torch.from_numpy(frame1).unsqueeze(dim=0)  # [H x W] -> [1 x H x W]

        if len(frame1.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            frame1 = np.expand_dims(frame1, -1)

        frame1 = np.moveaxis(frame1, -1, 0)  # H x W x C -> C x H x W
        frame1 = torch.from_numpy(frame1)

        # print('ft0, ft1 = ', frame0_timestamp + self.event_dataset.initial_stamp,
        #       frame1_timestamp + self.event_dataset.initial_stamp)
        # print('f_idx0, f_idx1 = ', frame0_idx, frame1_idx)

        if seed is None:
            # if no specific random seed was passed, generate our own.
            # otherwise, use the seed that was passed to us
            seed = random.randint(0, 2**32)

        if self.transform:
            random.seed(seed)
            frame0 = self.transform(frame0)
            random.seed(seed)
            frame1 = self.transform(frame1)

        # Get the event tensor from the event dataset loader
        # Note that we pass the transform seed to ensure the same transform is used
        events = self.event_dataset.__getitem__(i, seed)

        # Merge the 'frame' dictionary with the 'events' one
        item = {'frames': [frame0, frame1],
                **events}

        return item
