import numpy as np
import cnn_models
import linknet
from aux_metrics import jaccard_idx, sorensen_dice_coef
from visual_metrics import plot_history
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import backend as K
from keras.utils import multi_gpu_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

K.set_image_dim_ordering('tf')

#gpus=[0, 1, 2, 3]
BATCH_SIZE = 48 
EPOCHES = 256
VERBOSE = 1
OPTIMIZER = Adam() #SGD(lr=0.1)

print('\nLoading data')
X_train = np.load('../data/x_trn_1600_denom1_3bands.npy')
Y_train = np.load('../data/y_trn_1600_denom1_3bands.npy')
X_val = np.load('../data/x_eval_1600_denom1_3bands.npy')
Y_val = np.load('../data/y_eval_1600_denom1_3bands.npy')

print('Preprocessing data\n')
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255
X_train = np.transpose(X_train, (0, 2, 3, 1))
Y_train = np.transpose(Y_train, (0, 2, 3, 1))
X_val = np.transpose(X_val, (0, 2, 3, 1))
Y_val = np.transpose(Y_val, (0, 2, 3, 1))

model = cnn_models.UNet()
# load weights into new model
model.load_weights("../models/model.h5")
print("Loaded model from disk")

model.summary()
print('Model:   {}\n'.format(model.name))

parallel_model = model
#model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy', jaccard_idx, sorensen_dice_coef])
parallel_model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy', jaccard_idx, sorensen_dice_coef])

print('\nTest on validation samples')
score = parallel_model.evaluate(X_val, Y_val, verbose=VERBOSE)
print()
print('Test loss:        ', score[0])
print('Test accuracy:    ', score[1])
print('Jaccard index:    ', score[2])
print('Sorensen coef:    ', score[3])
print('DONE')
print()
