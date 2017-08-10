import numpy as np
import gc
import random
import pickle

from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils

from stream_data import data_set
from config import get_parameter

parameter = get_parameter()
root_path = parameter['root_path']
consecutive_frames = parameter['consecutive_frames']
img_rows = parameter['img_cols']
img_cols = parameter['img_cols']
num_classes = parameter['num_classes']


def cnn_spatial(lr):
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

    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.1)
    model = Model(inputs=model['input'], outputs=model['sofemax'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def train_spatial_model(class_index_dict, seed):
    batch_size = parameter['batch_size']
    iterations = parameter['iterations']

    # get data set
    print('Get data set.')
    # -------------------------------------------------------
    # training set
    training_set = data_set(class_index_dict, kind='train')

    # testing set
    # testing_set = get_data_set(class_index_dict, seed, 0, mini_batch=256, kind='test')
    testing_set = data_set(class_index_dict, kind='test')
    test_minibatch = testing_set.get_minibatch(seed, 0, mini_batch=256)
    X_test = np.array(test_minibatch['input']['spatial'])
    Y_test = np.array(test_minibatch['label'], dtype=np.float32)

    Y_test = np_utils.to_categorical(Y_test, num_classes)
    # -------------------------------------------------------
    print('Start training.')

    for i in range(iterations):
        print('%dth iterations' % (i + 1))

        # load model
        # ---------------------------------------------------------
        if i == 0:
            model = cnn_spatial(1e-2)
        if i == 14000:
            del model
            model = cnn_spatial(1e-3)
            model.load_weights(root_path + 'model/spatial_model')
            # ---------------------------------------------------------

        # get training data
        # -------------------------------------------------------
        # training_set = get_data_set(class_index_dict, seed, i, mini_batch=256, kind='train')
        train_minibatch = training_set.get_minibatch(seed, i, mini_batch=256)
        X_train = np.array(train_minibatch['input']['spatial'])
        Y_train = np.array(train_minibatch['label'], dtype=np.float32)

        Y_train = np_utils.to_categorical(Y_train, num_classes)
        # -------------------------------------------------------

        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=1,
                  # validation_data=(X_valid, Y_valid),
                  # validation_split=0.01, validation_data=None,
                  shuffle=True
                  # callbacks=[
                  #     EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto'),
                  #     ModelCheckpoint(root_path+'model/spatial_model', monitor='val_loss', verbose=0,
                  #                     save_best_only=False, save_weights_only=True, mode='auto', period=1)]
                  )

        # evaluate
        # -----------------------------------------------------------
        pred = model.predict(X_test, batch_size=batch_size)
        pred = np.array([np.argmax(i) for i in pred])
        print('accuracy: %.5f' % (np.sum(pred == Y_test) / len(Y_test)))
        # -----------------------------------------------------------

        model.save_weights(root_path + 'model/spatial_model', overwrite=True)

    # # Potentially save weights
    model.save_weights(root_path + 'model/spatial_model', overwrite=True)
    print('model saved.')

    # evaluate
    # -----------------------------------------------------------
    pred = model.predict(X_test, batch_size=batch_size)
    pred = np.array([np.argmax(i) for i in pred])
    print('accuracy: %.5f' % (np.sum(pred == Y_test) / len(Y_test)))
    # -----------------------------------------------------------

    del model, X_train, Y_train
    gc.collect()

if __name__ == "__main__":
    pickle_directory = parameter['pickle_directory']
    with open(pickle_directory + 'class_index_dict.pickle', 'rb') as fr:
        class_index_dict = pickle.load(fr)
    num_of_classes = len(class_index_dict) / 2
    seed = [random.random() for i in range(num_of_classes)]
    train_spatial_model(class_index_dict, seed)
