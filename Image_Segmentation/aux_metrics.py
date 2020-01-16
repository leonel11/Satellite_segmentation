from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

K.set_image_dim_ordering('tf')

SMOOTH = 1e-12

def jaccard_idx(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return K.mean(jac)

def sorensen_dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    sd = (2.0*intersection + SMOOTH) / (sum_ + SMOOTH)
    return K.mean(sd)

def dice_loss(y_true, y_pred):
    loss = 1 - sorensen_dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def focal_loss(y_true, y_pred, gamma=2.5, alpha=1.2):
    eps = 1e-6
    y_pred = K.clip(y_pred, eps, 1. - eps)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + eps)) - K.sum((1 - alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + eps))


def bce_focal_loss(y_true, y_pred):
    eps = 1e-6
    loss = binary_crossentropy(y_true, y_pred) + focal_loss(y_true, y_pred)*eps
    return loss

def balanced_cross_entropy(y_true, y_pred, beta=0.7):
    return -(beta*y_true*K.log(y_pred + SMOOTH) + (1.- beta)*(1. - y_true)*K.log(1. - y_pred + SMOOTH))

def focal_dice_loss(y_true, y_pred, gamma=2.0, alpha=0.75):
    eps = 1e-6
    loss = dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred, gamma, alpha)*eps
    return loss

def tversky_loss(y_true, y_pred, beta=0.7):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)