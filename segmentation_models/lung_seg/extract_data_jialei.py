import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
from os import walk, path

import SimpleITK as sitk
import cv2
import h5py
import imageio
import numpy as np
import pydicom
from skimage import transform, exposure


def preprocess_image(img, im_shape):
    img = np.squeeze(img)
    img = transform.resize(img, (im_shape, im_shape))
    img = exposure.equalize_hist(img)
    img -= img.mean()
    img /= img.std()
    img = np.expand_dims(img, -1)
    return img


def load_dataJSRT(group_name, path, im_shape):
    """This function loads data from hdf5 file"""
    X = []
    y = []
    z = []
    with h5py.File(path, "r") as f:
        group = f[group_name]
        img_names = group["img_ids"]
        images = group["img"]
        masks = group["mask"]

        for idx, name in enumerate(img_names):
            img = images[idx]
            X.append(preprocess_image(img, im_shape))

            mask = masks[idx]
            mask = np.squeeze(mask)
            mask = transform.resize(mask, (im_shape, im_shape)).astype(bool)
            mask = np.expand_dims(mask, -1)
            y.append(mask)

            z.append(name)

    X = np.array(X)
    y = np.array(y)
    z = np.array(z)

    print('### Data loaded')
    print('\t{}'.format(path))
    print('\t{}\t{}'.format(X.shape, y.shape))
    print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y, z


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
    # im = np.expand_dims(im, axis=0)
    return im

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
            img = norm(img)
            img = transform.resize(img, (im_shape, im_shape))
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


lung_dcm_path = 'D:/dataset/lungseg/lungseg_dcm'
lung_label_path = 'D:/dataset/lungseg/lungseg_label'

# get jialei labels
jialei_dcms = []
jialei_masks = []
jialei_fnames = []
for (dirpath, dirname, filenames) in walk(lung_label_path):
    label_paths = [path.join(dirpath, n) for n in filenames]
    for index, label in enumerate(label_paths):
        im = imageio.imread(label)
        im = np.array(im).sum(axis=2).astype(bool)
        im = transform.resize(im, (1024, 1024))
        jialei_masks.append(im)

        dcm_path = path.join(lung_dcm_path, filenames[index].strip('.png'))
        dim = load_dicom_2_image(dcm_path, 1024)
        jialei_dcms.append(dim)

        jialei_fnames.append(filenames[index].strip('.png'))

jialei_dcms = np.expand_dims(np.array(jialei_dcms), -1)
jialei_masks = np.expand_dims(np.array(jialei_masks), -1)
jialei_fnames = np.array(jialei_fnames)

# plot 4 pairs of data
for i in range(1, 5):
    index = i * 2 - 1
    plt.subplot(4, 2, index)
    x = jialei_dcms[i - 1, :, :, :]
    plt.imshow(x.squeeze())

    index = i * 2
    plt.subplot(4, 2, index)
    y = jialei_masks[i - 1, :, :, :]
    plt.imshow(y.squeeze())
plt.show()

# load jsrt dataset
x_jsrt_train, y_jsrt_train, id_jsrt_train = load_dataJSRT("train",
                                                          "C:/Users/edwardyangxin/Desktop/workprojs/tm_model/jsrt_1k.h5",
                                                          1024)
x_jsrt_test, y_jsrt_test, id_jsrt_test = load_dataJSRT("test",
                                                       "C:/Users/edwardyangxin/Desktop/workprojs/tm_model/jsrt_1k.h5",
                                                       1024)

# ensemble jialei and jsrt dataset
dcms = np.concatenate((jialei_dcms, x_jsrt_train, x_jsrt_test))
masks = np.concatenate((jialei_masks, y_jsrt_train, y_jsrt_test)).astype(bool)
fnames = np.concatenate((jialei_fnames, id_jsrt_train, id_jsrt_test))

# plot 4 pairs of data
for i in range(1, 5):
    index = i * 2 - 1
    plt.subplot(4, 2, index)
    x = dcms[i - 1, :, :, :]
    plt.imshow(x.squeeze())

    index = i * 2
    plt.subplot(4, 2, index)
    y = masks[i - 1, :, :, :]
    plt.imshow(y.squeeze())
plt.show()

# dcms = jialei_dcms
# masks = jialei_masks
# fnames = jialei_fnames
dataset_path = "../dataset/jia_jsrt_1k.hdf5"
# create h5 groups img\img_id\mask, datasets, add all data into one h5 file
with h5py.File(dataset_path, 'w') as h5f:
    count = dcms.shape[0]
    img_g = h5f.create_group("img")
    img_d = img_g.create_dataset('img', (count, 1024, 1024, 1), dtype="float32")

    dt = h5py.special_dtype(vlen=str)
    img_id_g = h5f.create_group("img_id")
    img_id_d = img_id_g.create_dataset('img_id', (count,), dtype=dt)

    mask_g = h5f.create_group("mask")
    mask_d = mask_g.create_dataset('mask', (count, 1024, 1024, 1), dtype="bool")

    for index, dcm in enumerate(dcms):
        img_d[index] = dcm
        img_id_d[index] = fnames[index]
        mask_d[index] = masks[index]

# check file and data
x, y = load_data(dataset_path, 256)

# plot 4 pairs of data
for i in range(1, 5):
    index = i * 2 - 1
    plt.subplot(4, 2, index)
    xx = x[i - 1, :, :]
    plt.imshow(xx.squeeze())

    index = i * 2
    plt.subplot(4, 2, index)
    yy = y[i - 1, :, :]
    plt.imshow(yy.squeeze())
plt.show()