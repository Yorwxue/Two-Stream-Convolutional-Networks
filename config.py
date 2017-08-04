# -*- coding:utf-8 -*-
from __future__ import print_function

import os

parameter = dict()
parameter['root_path'] = '/home/csist/clliao/workspace/Two-Stream-Convolutional-Networks/'  # /media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/two_stream_conv/
parameter['file_root_path'] = '/home/csist/clliao/dataset/'  # '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/dataset/'

parameter['consecutive_frames'] = 10  # signal as L in the paper
parameter['img_rows'] = 224
parameter['img_cols'] = 224
parameter['num_classes'] = 101
parameter['sample_freq_of_motion'] = 5

parameter['batch_size'] = 32
parameter['iterations'] = 20000

parameter['file_directory'] = parameter['file_root_path'] + 'UCF101/'
parameter['pickle_directory'] = parameter['file_root_path'] + 'pickle/'
parameter['index_directory'] = parameter['file_root_path'] + 'ucfTrainTestlist/'

parameter['split_selection'] = '01'  # '01', '02', '03'


def get_parameter():
    return parameter


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


def set_parameter():
    # if not os.path.exists(parameter['root_path']+'dataset'):
    #     os.mkdir('dataset')
    # if not os.path.exists(parameter['file_directory']):
    #     os.mkdir('dataset/UCF101')
    if not os.path.exists(parameter['pickle_directory']):
        os.mkdir(parameter['pickle_directory'])
    if not os.path.exists(parameter['index_directory']):
        os.mkdir(parameter['index_directory'])
    if not os.path.exists(parameter['root_path'] + 'model'):
        os.mkdir(parameter['root_path'] + 'model')


if __name__ == "__main__":
    set_parameter()
