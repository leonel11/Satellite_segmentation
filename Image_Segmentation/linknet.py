import numpy as np
from keras.models import Model
from keras.models import Input
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, concatenate
from keras.regularizers import l2
from keras import backend as K

np.random.seed(1)

IMG_CHANNELS = 3
IMG_ROWS = 160
IMG_COLS = 160
CLASSES = 3

K.set_image_dim_ordering('tf')


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return concatenate([shortcut, residual], axis=3)


def encoder_block(input_tensor, m, n):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
    added_1 = _shortcut(input_tensor, x)
    x = BatchNormalization()(added_1)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
    added_2 = _shortcut(added_1, x)
    return added_2


def decoder_block(input_tensor, m, n):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(1, 1))(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=(1, 1))(x)
    return x


def LinkNet(input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), classes=CLASSES):
    inputs = Input(shape=input_shape)
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    encoder_1 = encoder_block(input_tensor=x, m=64, n=64)
    encoder_2 = encoder_block(input_tensor=encoder_1, m=64, n=128)
    encoder_3 = encoder_block(input_tensor=encoder_2, m=128, n=256)
    decoder_3 = decoder_block(input_tensor=encoder_3, m=256, n=128)
    #encoder_4 = encoder_block(input_tensor=encoder_3, m=256, n=512)
    #decoder_4 = decoder_block(input_tensor=encoder_4, m=512, n=256)
    #decoder_3_in = concatenate([decoder_4, encoder_3], axis=3)
    #decoder_3_in = Activation('relu')(decoder_3_in)
    #decoder_3 = decoder_block(input_tensor=decoder_3_in, m=256, n=128)
    decoder_2_in = concatenate([decoder_3, encoder_2], axis=3)
    decoder_2_in = Activation('relu')(decoder_2_in)
    decoder_2 = decoder_block(input_tensor=decoder_2_in, m=128, n=64)
    decoder_1_in = concatenate([decoder_2, encoder_1], axis=3)
    decoder_1_in = Activation('relu')(decoder_1_in)
    decoder_1 = decoder_block(input_tensor=decoder_1_in, m=64, n=64)
    x = UpSampling2D((2, 2))(decoder_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=classes, kernel_size=(2, 2), padding="same")(x)
    model = Model(name='LinkNet', inputs=inputs, outputs=x)
    return model