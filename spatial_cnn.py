import numpy as np
import gc

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils

from stream_data import get_data_set
from config import get_parameter

parameter = get_parameter()
root_path = parameter['root_path']
consecutive_frames = parameter['consecutive_frames']
img_rows = parameter['img_cols']
img_cols = parameter['img_cols']
num_classes = parameter['num_classes']


def cnn_spatial():
    # img_rows = 224
    # img_cols = 224
    img_channels = 3
    input_shape = (img_rows, img_cols, img_channels)

    # build model
    model = dict()

    model['input'] = Input(shape=input_shape, dtype='float32')

    model['conv1'] = Conv2D(96, kernel_size=(7, 7), strides=2, activation='relu')(model['input'])
    model['conv1_norm'] = BatchNormalization()(model['conv1'])
    model['conv1_pool'] = MaxPooling2D(pool_size=(2, 2))(model['conv1_norm'])

    model['conv2'] = Conv2D(256, kernel_size=(5, 5), strides=2, activation='relu')(model['conv1_pool'])
    model['conv2_pool'] = MaxPooling2D(pool_size=(2, 2))(model['conv2'])

    model['conv3'] = Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu')(model['conv2_pool'])

    model['conv4'] = Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu')(model['conv3'])

    model['conv5'] = Conv2D(512, kernel_size=(3, 3), strides=1, activation='relu')(model['conv4'])
    model['conv5_pool'] = MaxPooling2D(pool_size=(2, 2))(model['conv5'])

    model['conv5_flat'] = Flatten()(model['conv5_pool'])

    model['full6'] = Dense(4096)(model['conv5_flat'])
    model['full6_drop'] = Dropout(0.9)(model['full6'])

    model['full7'] = Dense(2048)(model['full6_drop'])
    model['full7_drop'] = Dropout(0.9)(model['full7'])

    model['output'] = Dense(num_classes)(model['full7_drop'])

    model['sofemax'] = Activation('softmax')(model['output'])

    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.1)
    model = Model(inputs=model['input'], outputs=model['sofemax'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def train_spatial_model(training_set):
    mini_batch = 8  # 256
    epoch = 50
    model = cnn_spatial()

    # get training data
    if len(training_set) == 0:
        training_set, testing_set = get_data_set()
    X_train = np.array(training_set['input']['spatial'])
    Y_train = np.array(training_set['label'])

    Y_train = np_utils.to_categorical(Y_train, num_classes)

    ##############################################################################################
    print('Start training.')
    model.fit(X_train, Y_train,
              batch_size=mini_batch,
              epochs=epoch,
              # validation_data=(X_valid, Y_valid),
              validation_split=0.01, validation_data=None,
              shuffle=True,
              callbacks=[
                  EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto'),
                  ModelCheckpoint(root_path+'model/spatial_model', monitor='val_loss', verbose=0, save_best_only=False,
                                  save_weights_only=True, mode='auto', period=1)])
    #
    # # Potentially save weights
    model.save_weights(root_path+'model/spatial_model', overwrite=True)
    print('model saved.')
    ##############################################################################################
    del model, X_train, Y_train
    gc.collect()

if __name__ == "__main__":
    train_spatial_model([])
