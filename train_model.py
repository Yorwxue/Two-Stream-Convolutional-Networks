from __future__ import print_function

from keras import backend as K
import gc
import time

from temporal_cnn import train_temporal_model
from spatial_cnn import train_spatial_model
from stream_data import get_data_set


def train():
    training_set, _ = get_data_set()
    gc.collect()

    print('Training temporal model.')
    train_temporal_model(training_set)
    gc.collect()

    # ------------------------
    K.clear_session()
    # sess = tf.Session()
    # K.set_session(sess)
    # ------------------------

    print('Training spatial model.')
    train_spatial_model(training_set)
    gc.collect()

    print('ok.')

if __name__ == "__main__":
    train()
