# -*- coding:utf-8 -*-
from __future__ import print_function

parameter = dict()
parameter['root_path'] = '/media/clliao/006a3168-df49-4b0a-a874-891877a88870/clliao/workspace/python/two_stream_conv/'
parameter['consecutive_frames'] = 10  # signal as L in the paper
parameter['img_rows'] = 224
parameter['img_cols'] = 224
parameter['num_classes'] = 101


def get_parameter():
    return parameter


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')
