import ast
import csv

import h5py
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
from os import path

import SimpleITK as sitk
import cv2
import numpy as np
import pydicom
from PIL import Image, ImageDraw
from skimage import transform, exposure


def histeq(image):
    image_histogram, bins = np.histogram(image.flatten(), int(image.max() - image.min()), density=True)
    cdf = image_histogram.cumsum()
    cdf = cdf / cdf[-1]

    arr = np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape).astype(np.float32)
    return arr

def load_dicom_2_image(file, output_size, hist_eq=True):
    ds = pydicom.dcmread(file, stop_before_pixels=True)
    raw = sitk.ReadImage(file)
    im = sitk.GetArrayFromImage(raw).astype(np.float32)
    im = np.squeeze(im)
    height, width = im.shape
    if ds.PhotometricInterpretation == 'MONOCHROME1':
        im = im.max() - im

    if output_size / np.float(im.shape[0]) < 0.5:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    im = cv2.resize(src=im, dsize=(output_size,) * 2, interpolation=interp)
    if hist_eq:
        im = histeq(im)
    im -= im.mean()
    im /= np.std(im) + 1e-6
    return im, width, height

def norm(im):
    im -= im.mean()
    im /= np.std(im) + 1e-6
    return im

def load_data(path, im_shape):
    """This function loads data from hdf5 file"""
    X = []
    y = []
    with h5py.File(path, "r") as f:
        img_names = f["img_id/img_id"]
        images = f["img/img"]
        masks = f["mask/mask"]

        for idx, name in enumerate(img_names):
            img = images[idx]
            img = np.squeeze(img)
            img =cv2.resize(img, (im_shape, im_shape), interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, -1)
            X.append(img)

            mask = masks[idx]
            mask = np.squeeze(mask)
            mask = transform.resize(mask, (im_shape, im_shape))
            mask = np.expand_dims(mask, -1)
            y.append(mask)

    X = np.array(X)
    y = np.array(y)

    print('### Data loaded')
    print('\t{}'.format(path))
    print('\t{}\t{}'.format(X.shape, y.shape))
    print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y

file_prefix = "D:/dataset/lungseg/lungseg_anno_dcm_job1"
# format: filename, [[[nodes]],[[nodes]]]
anno_csv = 'C:/Users/edwardyangxin/Desktop/workprojs/segmentation_models/annotation_modified.csv'
width = 1024
check_all_mask = False
# get file mask pairs (file,mask)
anno_masks = []
anno_dcms = []
anno_fnames = []
with open(anno_csv, newline='') as csvfile:
    fs = csv.reader(csvfile, delimiter=',')
    for f in fs:
        poly_str = f[1]
        poly_list = ast.literal_eval(poly_str)
        if len(poly_list) == 2:
            # get dicom and info
            name = f[0]
            anno_fnames.append(name)

            dcm_path = path.join(file_prefix, name)
            dim, w, h = load_dicom_2_image(dcm_path, 1024)
            anno_dcms.append(dim)

            # get mask
            # left mask
            poly_l1 = poly_list[0]
            poly_l1 = [(coor[0]/512*width, coor[1]/(int(h*512/w))*width) for coor in poly_l1]
            img = Image.new('L', (width, width), 0)
            ImageDraw.Draw(img).polygon(poly_l1, outline=1, fill=1)
            left_mask = np.array(img)
            # right mask
            poly_l2 = poly_list[1]
            poly_l2 = [(coor[0]/512*width, coor[1]/(int(h*512/w))*width) for coor in poly_l2]
            img = Image.new('L', (width, width), 0)
            ImageDraw.Draw(img).polygon(poly_l2, outline=1, fill=1)
            right_mask = np.array(img)

            mask = left_mask + right_mask
            mask = mask.astype(bool)

            anno_masks.append(mask)
        else:
            print("skip not valid mask annotation:{}".format(name))

anno_dcms = np.expand_dims(np.array(anno_dcms), -1)
anno_masks = np.expand_dims(np.array(anno_masks), -1)
anno_fnames = np.array(anno_fnames)

# plot 4 pairs of data
for i in range(1, 5):
    plt.subplot(4, 1, i)
    x = anno_dcms[i - 3, :, :, :]
    plt.imshow(x.squeeze(), cmap='gray', alpha=1)
    y = anno_masks[i - 3, :, :, :]
    plt.imshow(y.squeeze().astype(int)*100, cmap='Oranges', alpha=0.4)
plt.show()

if check_all_mask:
    for i, dcm in enumerate(anno_dcms):
        plt.figure()
        plt.imshow(dcm.squeeze(), cmap='gray', alpha=1)
        y = anno_masks[i, :, :, :]
        plt.imshow(y.squeeze().astype(int) * 100, cmap='Oranges', alpha=0.4)
        plt.savefig("D:/dataset/lungseg/lungseg_anno_checkmask_job1/{}.png".format(anno_fnames[i]))

dataset_path = "../dataset/anno_lungseg_1k.hdf5"
# create h5 groups img\img_id\mask, datasets, add all data into one h5 file
with h5py.File(dataset_path, 'w') as h5f:
    count = anno_dcms.shape[0]
    img_g = h5f.create_group("img")
    img_d = img_g.create_dataset('img', (count, 1024, 1024, 1), dtype="float32")

    dt = h5py.special_dtype(vlen=str)
    img_id_g = h5f.create_group("img_id")
    img_id_d = img_id_g.create_dataset('img_id', (count,), dtype=dt)

    mask_g = h5f.create_group("mask")
    mask_d = mask_g.create_dataset('mask', (count, 1024, 1024, 1), dtype="bool")

    for index, dcm in enumerate(anno_dcms):
        img_d[index] = dcm
        img_id_d[index] = anno_fnames[index]
        mask_d[index] = anno_masks[index]

# check file and data
x, y = load_data(dataset_path, 256)

# plot 4 pairs of data
for i in range(1, 5):
    plt.subplot(4, 1, i)
    xx = x[i - 1, :, :, ]
    plt.imshow(xx.squeeze(), cmap='gray', alpha=1)
    yy = y[i - 1, :, :, :]
    plt.imshow(yy.squeeze().astype(int)*100, cmap='Oranges', alpha=0.4)
plt.show()
