import glob
import argparse
import os
import h5py
import pandas as pd
import numpy as np
from zipfile import ZipFile
import cv2
from tqdm import tqdm
from event_packagers import *


def get_sensor_size(zipdata):
    try:
        for imfile in zipdata.namelist():
            if 'frame_' in imfile:
                break
        img = cv2.imdecode(np.frombuffer(zipdata.read(imfile), np.uint8), cv2.IMREAD_GRAYSCALE)
        sensor_size = img.shape[:2]
    except:
        sensor_size = None
        print('Warning: could not read sensor size')
    return sensor_size


def extract_zip(zip_path, output_path, zero_timestamps=False,
                packager=hdf5_packager):
    ep = packager(output_path)
    first_ts = -1
    t0 = -1
    if not os.path.exists(zip_path):
        print("{} does not exist!".format(zip_path))
        return

    zipdata = ZipFile(zip_path, 'r')
    # compute sensor size
    sensor_size = get_sensor_size(zipdata)
    # Extract events to h5
    imgfiles = [x for x in zipdata.namelist() if 'frame' in x]
    evtfile, imgtsfile = None, None
    for file in zipdata.namelist():
        if 'events' in file:
            evtfile = file
        if 'images.txt' in file:
            imgtsfile = file
        if evtfile and imgtsfile:
            break
    ep.set_data_available(num_images=len(imgfiles), num_flow=0)
    total_num_pos, total_num_neg, last_ts = 0, 0, 0

    print("packing images ...")
    chunksize = 1
    iterator = pd.read_csv(zipdata.open(imgtsfile), delim_whitespace=True, header=None,
                           names=['t', 'imfile'],
                           dtype={'t': np.float64, 'imfile': str},
                           engine='c',
                           skiprows=0, chunksize=chunksize, nrows=None, memory_map=True)
    for i, imdatas in tqdm(enumerate(iterator)):
        ts, _ = imdatas.values[0]
        imfile = imgfiles[i]
        img = cv2.imdecode(np.frombuffer(zipdata.read(imfile), np.uint8), cv2.IMREAD_GRAYSCALE)
        img_id = i
        if first_ts == -1:
            first_ts = ts
        if zero_timestamps:
            ts -= first_ts

        ep.package_image(img, ts, img_id)

    print("packing events ...")
    first_ts = -1
    chunksize = 1000000
    iterator = pd.read_csv(zipdata.open(evtfile), delim_whitespace=True, header=None,
                           names=['t', 'x', 'y', 'pol'],
                           dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                           engine='c',
                           skiprows=0, chunksize=chunksize, nrows=None, memory_map=True)

    pbar = tqdm(enumerate(iterator))
    for i, event_window in pbar:
        events = event_window.values
        ts = events[:, 0].astype(np.float64)
        xs = events[:, 1].astype(np.int16)
        ys = events[:, 2].astype(np.int16)
        ps = events[:, 3]
        ps[ps < 0] = 0 # should be [0 or 1]
        ps = ps.astype(bool)

        if first_ts == -1:
            first_ts = ts[0]

        if zero_timestamps:
            ts -= first_ts
        last_ts = ts[-1]
        if sensor_size is None or sensor_size[0] < max(ys) or sensor_size[1] < max(xs):
            sensor_size = [max(xs), max(ys)]
            pbar.set_description("Sensor size inferred from events as {}".format(sensor_size))

        sum_ps = sum(ps)
        total_num_pos += sum_ps
        total_num_neg += len(ps) - sum_ps
        ep.package_events(xs, ys, ts, ps)
        if i % 10 == 9:
            pbar.set_description('Events written: {} M'.format((total_num_pos + total_num_neg) / 1e6))
    print('Events written: {} M'.format((total_num_pos + total_num_neg) / 1e6))
    print("Detect sensor size {}".format(sensor_size))
    t0 = 0 if zero_timestamps else first_ts
    ep.add_metadata(total_num_pos, total_num_neg, last_ts-t0, t0, last_ts, num_imgs=len(imgfiles), num_flow=0, sensor_size=sensor_size)


def extract_zips(zip_paths, output_dir, zero_timestamps=False):
    for path in zip_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, "{}.h5".format(filename))
        print("Extracting {} to {}".format(path, out_path))
        extract_zip(path, out_path, zero_timestamps=zero_timestamps)


if __name__ == "__main__":
    """
    Tool for converting txt events to an efficient HDF5 format that can be speedily
    accessed by python code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/media/gph/FILE/Data/IJRR/eval", help="zip file to extract or directory containing txt files")
    parser.add_argument("--output_dir", default="/media/gph/FILE/Data/IJRR/eval", help="Folder where to extract the data")
    parser.add_argument('--zero_timestamps', default=True, help='If true, timestamps will be offset to start at 0')
    args = parser.parse_args()

    print('Data will be extracted in folder: {}'.format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isdir(args.path):
        zip_paths = sorted(list(glob.glob(os.path.join(args.path, "*.zip"))))
    else:
        zip_paths = [args.path]
    extract_zips(zip_paths, args.output_dir, zero_timestamps=args.zero_timestamps)
