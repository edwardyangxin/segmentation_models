#!/usr/bin/env python3
import argparse
from os.path import join, isfile
from os import listdir

from sklearn.model_selection import train_test_split
import h5py
import numpy as np
from skimage import io
import cv2

def list_files(dir, suffix=None):
    if suffix:
        return [f for f in listdir(dir) if isfile(join(dir, f))]
    else:
        return [f for f in listdir(dir) if f.endswith(suffix) and isfile(join(dir, f))]


def check_data_integrity(img_ids, left_masks, right_masks):
    left_mask_ids = [f[:-4] for f in left_masks]
    right_mask_ids = [f[:-4] for f in right_masks]
    is_intergrity = True
    
    for id in img_ids:
        if id not in left_mask_ids:
            print("{}.IMG left mask image misssing.")
            is_intergrity = False
        
        if id not in right_mask_ids:
            print("{}.IMG right mask image misssing.")
            is_intergrity = False
    return is_intergrity


def read_img(file):
    try:
        raw = np.fromfile(file, dtype=">i2").astype("<i2").reshape((2048,2048))
        raw = raw.max() - raw
        return cv2.resize(raw, dsize=(1024,1024), interpolation=cv2.INTER_CUBIC)
    except Exception as identifier:
        print(file)
        print(identifier)


def read_mask(left_mask_file, right_mask_file):
    left_mask = io.imread(left_mask_file)
    right_mask = io.imread(right_mask_file)
    return left_mask + right_mask



def load_dataset(img_ids, args, group):
    img_dset = group.create_dataset("img", (len(img_ids), 1024, 1024, 1), dtype="u2")
    mask_dset = group.create_dataset("mask", (len(img_ids), 1024, 1024, 1), dtype=np.bool)
    
    for idx, img_id in enumerate(img_ids):
        img = read_img(join(args.img_dir, img_id + ".IMG"))
        mask = read_mask(
            join(args.left_mask_dir, img_id + '.gif'),
            join(args.right_mask_dir, img_id + '.gif')
        )
        img_dset[idx] = np.expand_dims(img, axis=2)
        mask_dset[idx] = np.expand_dims(mask, axis=2).astype(np.bool)


def pack_jsrt_dataset(args):
    img_files = list_files(args.img_dir, ".IMG")
    left_mask_files = list_files(args.left_mask_dir, "gif")
    right_mask_files = list_files(args.right_mask_dir, "gif")

    img_ids = [f[:-4] for f in img_files]
    if not check_data_integrity(img_ids, left_mask_files, right_mask_files):
        exit(0)
    
    train_set, test_set = train_test_split(img_ids, test_size=0.2, random_state=42)
    print('total data size: %d'%len(img_ids))
    print('train data size: %d'%len(train_set))
    print('test data size: %d'%len(test_set))
    
    h5file = h5py.File(args.output, 'w')
    train_group = h5file.create_group("train")
    test_group = h5file.create_group("test")
    
    dt = h5py.special_dtype(vlen=str)
    dset = train_group.create_dataset("img_ids", (len(train_set), ), dtype=dt)
    for idx, id in enumerate(train_set):
        dset[idx] = id
    
    dset = test_group.create_dataset("img_ids", (len(test_set), ),dtype=dt)
    for idx, id, in enumerate(test_set):
        dset[idx] = id

    print("Packing train set...")
    load_dataset(train_set, args, train_group)
    print("Packing test set...")
    load_dataset(test_set, args, test_group)
    h5file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="JSRT DataSet Packer", description="Pack JSRT dataset in hdf5 data format")
    parser.add_argument("--img-dir", type=str, help="Image data directory")
    parser.add_argument("--left-mask-dir", type=str, help="left lung mask image diretory")
    parser.add_argument("--right-mask-dir", type=str, help="right lung mask image directory")
    parser.add_argument("--output", type=str, help="output diretory")
    args = parser.parse_args()
    pack_jsrt_dataset(args)
