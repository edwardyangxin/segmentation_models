import imageio
import numpy as np
from os import walk, path

import pydicom
import SimpleITK as sitk
import cv2


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


lung_dcm_path = 'D:/dataset/lungseg/lungseg_dcm'
lung_label_path = 'D:/dataset/lungseg/lungseg_label'

# get jialei labels
dcms = []
labels = []
for (dirpath, dirname, filenames) in walk(lung_label_path):
    label_paths = [path.join(dirpath, n) for n in filenames[:3]]
    for index, label in enumerate(label_paths):
        im = imageio.imread(label)
        im = np.array(im).sum(axis=2).astype(bool).astype(int)
        labels.append(im)

        dcm_path = path.join(lung_dcm_path, filenames[index].strip('.png'))
        dim = load_dicom_2_image(dcm_path, 1024)
        dcms.append(dim)
dcms = np.array(dcms)
labels = np.array(labels)

# ensemble jialei and jsrt dataset




