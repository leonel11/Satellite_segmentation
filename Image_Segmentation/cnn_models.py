import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.models import Input
from keras.layers import concatenate, BatchNormalization, Activation, Lambda, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Reshape, Permute
from keras import backend as K
from aux_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

np.random.seed(1)

K.set_image_dim_ordering('tf')

IMG_CHANNELS = 3
IMG_ROWS = 160
IMG_COLS = 160
CLASSES = 3

def UNet():
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    conv1 = Conv2D(32, kernel_size=3, padding='same')(inputs)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, kernel_size=3, padding='same')(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=3, padding='same')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, kernel_size=3, padding='same')(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=3, padding='same')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, kernel_size=3, padding='same')(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, kernel_size=3, padding='same')(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(256, kernel_size=3, padding='same')(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, kernel_size=3, padding='same')(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, kernel_size=3, padding='same')(conv5)
    conv5 = Activation('relu')(conv5)
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(256, kernel_size=3, padding='same')(merge6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, kernel_size=3, padding='same')(conv6)
    conv6 = Activation('relu')(conv6)
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(128, kernel_size=3, padding='same')(merge7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, kernel_size=3, padding='same')(conv7)
    conv7 = Activation('relu')(conv7)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(64, kernel_size=3, padding='same')(merge8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, kernel_size=3, padding='same')(conv8)
    conv8 = Activation('relu')(conv8)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(32, kernel_size=3, padding='same')(merge9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(32,kernel_size=3, padding='same')(conv9)
    conv9 = Activation('relu')(conv9)
    outputs = Conv2D(CLASSES, kernel_size=1, activation='sigmoid')(conv9)
    model = Model(name='UNet', inputs=inputs, outputs=outputs)
    return model


def SegNet():
    # encoder
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    conv1 = Conv2D(64, kernel_size=3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(64, kernel_size=3, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool1, mask1 = MaxPoolingWithArgmax2D((2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=3, padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv4 = Conv2D(128, kernel_size=3, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    pool2, mask2 = MaxPoolingWithArgmax2D((2, 2))(conv4)
    conv5 = Conv2D(256, kernel_size=3, padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv6 = Conv2D(256, kernel_size=3, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv7 = Conv2D(256, kernel_size=3, padding='same')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    pool3, mask3 = MaxPoolingWithArgmax2D((2, 2))(conv7)
    conv8 = Conv2D(512, kernel_size=3, padding='same')(pool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)
    conv9 = Conv2D(512, kernel_size=3, padding='same')(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)
    conv10 = Conv2D(512, kernel_size=3, padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation("relu")(conv10)
    pool4, mask4 = MaxPoolingWithArgmax2D((2, 2))(conv10)
    conv11 = Conv2D(512, kernel_size=3, padding='same')(pool4)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation("relu")(conv11)
    conv12 = Conv2D(512, kernel_size=3, padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation("relu")(conv12)
    conv13 = Conv2D(512, kernel_size=3, padding='same')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation("relu")(conv13)
    pool5, mask5 = MaxPoolingWithArgmax2D((2, 2))(conv13)
    # decoder
    unpool1 = MaxUnpooling2D((2, 2))([pool5, mask5])
    conv14 = Conv2D(512, kernel_size=3, padding='same')(unpool1)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    conv15 = Conv2D(512, kernel_size=3, padding='same')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)
    conv16 = Conv2D(512, kernel_size=3, padding='same')(conv15)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    unpool2 = MaxUnpooling2D((2, 2))([conv16, mask4])
    conv17 = Conv2D(512, kernel_size=3, padding='same')(unpool2)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    conv18 = Conv2D(512, kernel_size=3, padding='same')(conv17)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)
    conv19 = Conv2D(256, kernel_size=3, padding='same')(conv18)
    conv19 = BatchNormalization()(conv19)
    conv19 = Activation('relu')(conv19)
    unpool3 = MaxUnpooling2D((2, 2))([conv19, mask3])
    conv20 = Conv2D(256, kernel_size=3, padding='same')(unpool3)
    conv20 = BatchNormalization()(conv20)
    conv20 = Activation('relu')(conv20)
    conv21 = Conv2D(256, kernel_size=3, padding='same')(conv20)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    conv22 = Conv2D(128, kernel_size=3, padding='same')(conv21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation('relu')(conv22)
    unpool4 = MaxUnpooling2D((2, 2))([conv22, mask2])
    conv23 = Conv2D(128, kernel_size=3, padding='same')(unpool4)
    conv23 = BatchNormalization()(conv23)
    conv23 = Activation('relu')(conv23)
    conv24 = Conv2D(64, kernel_size=3, padding='same')(conv23)
    conv24 = BatchNormalization()(conv24)
    conv24 = Activation('relu')(conv24)
    unpool5 = MaxUnpooling2D((2, 2))([conv24, mask1])
    conv25 = Conv2D(64, kernel_size=3, padding='same')(unpool5)
    conv25 = BatchNormalization()(conv25)
    conv25 = Activation('relu')(conv25)
    conv26 = Conv2D(CLASSES, kernel_size=1, padding="valid")(conv25)
    conv26 = BatchNormalization()(conv26)
    #conv26 = Reshape((CLASSES, IMG_ROWS * IMG_COLS))(conv26)
    #conv26 = Permute((2, 1))(conv26)
    outputs = Activation('softmax')(conv26)
    segnet = Model(name='SegNet', inputs=inputs, outputs=outputs)
    return segnet


def SegNetBasic():
    encoding_layers = [
        Conv2D(64, kernel_size=3, input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]
    autoencoder = Sequential()
    autoencoder.encoding_layers = encoding_layers
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    decoding_layers = [
        UpSampling2D(),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(CLASSES, kernel_size=1, padding='valid', activation='sigmoid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    #autoencoder.add(Reshape((CLASSES, IMG_ROWS * IMG_COLS)))
    #autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    autoencoder.name = 'SegNetBasic'
    return autoencoder
