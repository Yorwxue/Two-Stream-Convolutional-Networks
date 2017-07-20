from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Model


def cnn_temporal():
    img_rows = 224
    img_cols = 224
    input_volume = 1
    img_channels = 2 * input_volume
    mini_batch = 256
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
    model['full6_drop'] = Dropout(0.7)(model['full6'])

    model['full7'] = Dense(2048)(model['full6_drop'])
    model['full7_drop'] = Dropout(0.8)(model['full7'])

    model['sofemax'] = Activation('softmax')(model['full7_drop'])

    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.1)
    model = Model(inputs=model['input'], outputs=model['sofemax'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd)


if __name__ == "__main__":
    cnn_temporal()
