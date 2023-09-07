import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm
from event_packagers import *


def append_to_dataset(dataset, data):
    dataset.resize(dataset.shape[0] + len(data), axis=0)
    if len(data) == 0:
        return
    dataset[-len(data):] = data[:]


def timestamp_float(ts):
    return ts.secs + ts.nsecs / float(1e9)


# Inspired by https://github.com/uzh-rpg/rpg_e2vid
def cut_h5file(h5file, output_path, out_name_suffix, t0, tk,
               packager=hdf5_packager, sensor_size=None):
    filename = os.path.split(h5file)[-1].split('.')[0]
    out_file = os.path.join(output_path, f"{filename}{out_name_suffix}.h5")
    ep = packager(out_file)

    if not os.path.exists(h5file):
        print("{} does not exist!".format(h5file))
        return
    with h5py.File(h5file, 'r') as h5_file:
        t0_file, tk_file = h5_file.attrs['t0'], h5_file.attrs['tk']
        assert t0 < tk_file and tk > t0_file, F"Wrong value for t0/tk(wanted:{t0}/{tk}, exits:{t0_file}/{tk_file})"
        # Look for the topics that are available and save the total number of messages for each topic (useful for the progress bar)

        # Extract events to h5
        print('Extracting events ...')
        ts = h5_file['events/ts'][()]
        ind = np.where(np.logical_and(ts >= t0, ts <= tk))[0]
        minId, maxId = min(ind), max(ind)
        xs, ys, ts, ps = h5_file['events/xs'][minId: maxId],\
                         h5_file['events/ys'][minId: maxId],\
                         h5_file['events/ts'][minId: maxId], \
                         h5_file['events/ps'][minId: maxId]
        ep.package_events(xs, ys, ts, ps)
        num_pos = sum(ps)
        num_neg = len(ps) - num_pos
        start_time, end_time = ts[0], ts[-1]
        duration = end_time - start_time

        # Extract images to h5
        print('Extracting images ...')
        imdatainfos = [(k, v.attrs['timestamp']) for k, v in h5_file['images'].items()
                       if v.attrs['timestamp'] >= t0 and v.attrs['timestamp'] <= tk]
        num_imgs = len(imdatainfos)
        ep.set_data_available(num_imgs, 0)
        for i, data in enumerate(imdatainfos):
            name, timestamp = data
            image = h5_file[f"images/{name}"][()]
            ep.package_image(image, timestamp, i)
            sensor_size = image.shape
        print("Detect sensor size {}".format(sensor_size))
        ep.add_metadata(num_pos, num_neg, duration, start_time, end_time, num_imgs, num_flow=0, sensor_size=sensor_size)


def cut_h5files(h5file_txt, output_dir, suffix):
    dataRoot = os.path.split(h5file_txt)[0]
    with open(h5file_txt, 'r') as fp:
        datas = fp.readlines()
    for data in datas:
        t0, tk, file = data.split()
        t0, tk = float(t0), float(tk)
        file = os.path.join(dataRoot, file)
        cut_h5file(file, output_dir, suffix, t0, tk)


if __name__ == "__main__":
    """
    Tool for converting rosbag events to an efficient HDF5 format that can be speedily
    accessed by python code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/media/gph/FILE/Data/MVSEC/rosbag/eval/cut.txt", help="h5 file to cut or txtfile indicating h5 files")
    parser.add_argument("--output_dir", default="/media/gph/FILE/Data/MVSEC/rosbag/eval", help="Folder where to save the data")
    parser.add_argument("--suffix", default="_cut", help="")
    parser.add_argument("--t0", default=0.0)
    parser.add_argument("--tk", default=0.0)
    args = parser.parse_args()

    print('Data will be extracted in folder: {} with suffix {}'.format(args.output_dir, args.suffix))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cut_h5files(args.path, args.output_dir, args.suffix)


