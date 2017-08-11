from __future__ import print_function

from keras import backend as K
import gc
import random
import pickle

from temporal_cnn import train_temporal_model
from spatial_cnn import train_spatial_model
from config import get_parameter
from stream_data import stack_optical_flow

parameter = get_parameter()
pickle_directory = parameter['pickle_directory']
file_directory = parameter['file_directory']


def train():
    stack_optical_flow(file_directory, data_update=False)
    with open(pickle_directory + 'class_index_dict.pickle', 'rb') as fr:
        class_index_dict = pickle.load(fr)
    num_of_classes = int(len(class_index_dict) / 2)
    # seed = [random.random() for i in range(num_of_classes)]

    print('Training temporal model.')
    train_temporal_model(class_index_dict)
    gc.collect()

    # release memory
    # ------------------------
    K.clear_session()
    # sess = tf.Session()
    # K.set_session(sess)
    # ------------------------

    # print('Training spatial model.')
    # train_spatial_model(class_index_dict)
    # gc.collect()

    print('ok.')

if __name__ == "__main__":
    train()
