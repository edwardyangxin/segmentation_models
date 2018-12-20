import h5py
import numpy as np
from skimage import transform, exposure

from segmentation_models import Unet


def preprocess_image(img, im_shape):
    img = np.squeeze(img)
    img = transform.resize(img, (im_shape, im_shape))
    img = exposure.equalize_hist(img)
    img -= img.mean()
    img /= img.std()
    # img = np.expand_dims(img, -1)
    return img

def load_dataJSRT(group_name, path, im_shape):
    """This function loads data from hdf5 file"""
    X = []
    y = []
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


# prepare data
x, y = load_dataJSRT("train", "C:/Users/edwardyangxin/Desktop/workprojs/tm_model/jsrt_1k.h5", 256)
x = np.stack([x]*3, axis=-1)

# prepare model
model = Unet(backbone_name='resnet34', input_shape=(256, 256, 3), encoder_weights='imagenet')
model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# train model
model.fit(x, y)