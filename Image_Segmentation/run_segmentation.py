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
from aux_metrics import bce_dice_loss, focal_loss, bce_focal_loss, balanced_cross_entropy, focal_dice_loss, tversky_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

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
model.summary()
print('Model:   {}\n'.format(model.name))
#for g in gpus:
#    with K.tf.device('/gpu:{}'.format(g)):
#        kernel_model = model
#parallel_model = multi_gpu_model(kernel_model, gpus=len(gpus))
parallel_model = model
#model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy', jaccard_idx, sorensen_dice_coef])
#parallel_model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy', jaccard_idx, sorensen_dice_coef])
parallel_model.compile(optimizer=OPTIMIZER, loss=tversky_loss, metrics=['accuracy', jaccard_idx, sorensen_dice_coef])
history = parallel_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHES, verbose=VERBOSE,
                    validation_data=(X_val, Y_val))
parallel_model.save_weights("../models/model.h5")
print("Saved model to disk")
print('\nTest on validation samples')
score = parallel_model.evaluate(X_val, Y_val, verbose=VERBOSE)
print()
print('Test loss:        ', score[0])
print('Test accuracy:    ', score[1])
print('Jaccard index:    ', score[2])
print('Sorensen coef:    ', score[3])
print('\nPlot history\n')
plot_history(history, model.name, EPOCHES)
print('DONE')
print()
