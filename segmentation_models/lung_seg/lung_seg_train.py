import h5py
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from skimage import transform
from keras.callbacks import ModelCheckpoint
from segmentation_models import Unet
from segmentation_models.lung_seg.image_generator_keras import ImageDataGenerator


def load_data(path, im_shape):
    """This function loads data from hdf5 file"""
    X = []
    y = []
    with h5py.File(path, "r") as f:
        img_names = f["img_id/img"]
        images = f["img/img"]
        masks = f["mask/mask"]

        for idx, name in enumerate(img_names):
            img = images[idx]
            img = np.squeeze(img)
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


def split_data(x, y, index, td, vd):
    count = x.shape[index]
    tr = count * td // 10
    val = count * vd // 10
    return x[0:tr], y[0:tr], x[tr:tr + val], y[tr:tr + val], x[tr + val:], y[tr + val:]


# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


# train:validation:test, 7:2:1 here
traind = 7
vald = 2

# prepare data
x, y = load_data(
    "C:/Users/edwardyangxin/Desktop/workprojs/segmentation_models/segmentation_models/dataset/jia_jsrt_1k.hdf5", 256)
x = x[:100]
x = np.stack([x] * 3, axis=-1)
x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y, 0, traind, vald)

# prepare model
model = Unet(backbone_name='resnet34', input_shape=(256, 256, 3), encoder_weights='imagenet')
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[mean_iou, 'binary_accuracy'])

model_file_format = 'weights.{epoch:02d}-{val_acc:.3f}.hdf5'
checkpointer = ModelCheckpoint(model_file_format, monitor='val_mean_iou', period=1)

train_gen = ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1.,
                               zoom_range=0.2,
                               fill_mode='nearest',
                               cval=0)
test_gen = ImageDataGenerator(rescale=1.)
# train model
batch_size = 8
model.fit_generator(train_gen.flow(x_train, y_train, batch_size),
                    steps_per_epoch=(x_train.shape[0] + batch_size - 1) // batch_size,
                    epochs=100,
                    callbacks=[checkpointer],
                    validation_data=test_gen.flow(x_val, y_val),
                    validation_steps=(x_val.shape[0] + batch_size - 1) // batch_size)
