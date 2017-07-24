# -*- coding: utf-8 -*-
from __future__ import print_function

import cPickle
import numpy as np
import os
import cv2
import tqdm
import time

from config import get_parameter, time_spent_printer

parameter = get_parameter()
root_path = parameter['root_path']
consecutive_frames = parameter['consecutive_frames']
img_rows = parameter['img_cols']
img_cols = parameter['img_cols']

file_directory = root_path + 'dataset/UCF101/'
pickle_directory = root_path + 'dataset/pickle/'
# file_name = 'v_Archery_g01_c07.avi'
# file_name = 'v_Skiing_g21_c01.avi'
index_directory = root_path + 'dataset/ucfTrainTestlist/'

data_update = True  # True / False

test_program_flag = 100  # only for testing program


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)  # 以網格的形式選取二维圖像上等間隔的點，這裡間隔为16，reshape成2行的array
    fx, fy = flow[y, x].T  # 取選定網格點座標對應的光流位移
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)  # 将初始點和變化的點堆疊成2*2的數組
    lines = np.int32(lines + 0.5)  # 忽略微笑的假偏移，整數化
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))  # 以初始點和終點劃線表示光流運動
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # 在初始點（網格點處畫圆點来表示初始點）
    return vis


def create_optical_flow(video_file, sample_freq_of_motion=5):
    cap = cv2.VideoCapture(video_file)

    frame_set = dict()
    frame_set['orig'] = list()
    frame_set['gray'] = list()
    frame_set['flow'] = list()
    frame_set['hori'] = list()
    frame_set['vert'] = list()

    count = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:  # end/pause of this video
            # exit()
            break

        # Operations on the frame come here
        # ----------------------------------------------------------------------
        frame_set['orig'].append(cv2.resize(frame, (img_rows, img_cols)))
        frame_set['gray'].append(cv2.cvtColor(frame_set['orig'][count], cv2.COLOR_BGR2GRAY))

        # calculate motion of two frames
        if ((count % sample_freq_of_motion) == 0) and (count != 0):
            frame_set['flow'].append(cv2.calcOpticalFlowFarneback(frame_set['gray'][count-sample_freq_of_motion],
                                                                  frame_set['gray'][count],
                                                                  None, 0.5, 3, 15, 3, 5, 1.2, 0))
            # Display flow
            # cv2.imshow('frame', draw_flow(frame_set['gray'][count], frame_set['flow'][-1]))
            # cv2.waitKey(10)

            # horizontal & vertical
            # frame_set['hori'].append(frame_set['flow'][-1][..., 0])
            frame_set['hori'].append(frame_set['flow'][-1][..., 0] - cv2.mean(frame_set['flow'][-1][..., 0])[0])
            # frame_set['vert'].append(frame_set['flow'][-1][..., 1])
            frame_set['vert'].append(frame_set['flow'][-1][..., 1] - cv2.mean(frame_set['flow'][-1][..., 1])[0])

            # change range to 0~255
            # frame_set['hori'][-1] = cv2.normalize(frame_set['hori'][-1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            # frame_set['vert'][-1] = cv2.normalize(frame_set['vert'][-1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

            # Display flow
            # cv2.imshow('frame', frame_set['hori'][-1])
            # cv2.imshow('frame', frame_set['vert'][-1])
            # cv2.waitKey(25)
        # ----------------------------------------------------------------------

        # Display the resulting frame
        # cv2.imshow('frame', frame_set['orig'][count])
        # cv2.waitKey(10)

        count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return frame_set


def stack_optical_flow(file_directory):

    # -- start from pickle --
    if os.path.exists(pickle_directory+'train.pickle') and not data_update:
        start_time = time.time()
        print('Read pickle files .. ', end='')

        fr = open(pickle_directory+'train.pickle', 'rb')
        training_set = cPickle.load(fr)
        fr.close()

        fr = open(pickle_directory + 'test.pickle', 'rb')
        testing_set = cPickle.load(fr)
        fr.close()

        print('ok')
        final_time = time.time()
        time_spent_printer(start_time, final_time)
        return training_set, testing_set

    # -- start from raw data --
    training_set = dict()
    testing_set = dict()

    training_set['input'] = dict()
    training_set['input']['temporal'] = list()
    training_set['input']['spatial'] = list()

    training_set['label'] = list()

    testing_set['input'] = dict()
    testing_set['input']['temporal'] = list()
    testing_set['input']['spatial'] = list()

    # training/testing data split
    # ----------------------------------------------------------------------
    print('Split training set and testing set.')
    training_index = dict()
    testing_index = dict()

    training_index['name'] = list()
    training_index['label'] = list()
    testing_index['name'] = list()

    training_files = ['trainlist01.txt', 'trainlist02.txt', 'trainlist03.txt']
    testing_files = ['testlist01.txt', 'testlist02.txt', 'testlist03.txt']

    for training_file in training_files:
        with open(index_directory+'%s' % training_file, 'r') as fr:
            lines = fr.readlines()
            training_index['name'] += [entry.split(' ')[0][entry.split(' ')[0].index('/')+1:] for entry in lines]
            training_index['label'] += [entry.split(' ')[1].replace('\r\n', '') for entry in lines]

    for testing_file in testing_files:
        with open(index_directory+'%s' % testing_file, 'r') as fr:
            lines = fr.readlines()
            testing_index['name'] += [entry[entry.index('/')+1:].replace('\r\n', '') for entry in lines]

    # ----------------------------------------------------------------------

    # make a list to record all videos in this directory
    # -------------------------------------------------
    print('Prepare input data and data label.')
    if os.path.exists(file_directory):
        if os.path.isdir(file_directory):
            file_list = os.listdir(file_directory)
            num_of_files = len(file_list)

            # for file_name in file_list:
            for file_index in tqdm.tqdm(range(num_of_files)):  # test_program_flag
                file_name = file_list[file_index]

                # Allocate input data to temporal and spatial set
                # ------------------------------------------------
                def allocate_input_data():
                    index = 0
                    temporal_input_list = list()
                    spatial_input_list = list()

                    frame_input = create_optical_flow(file_directory + file_name)
                    frame_input_len = np.min([len(frame_input['hori']), len(frame_input['vert'])])
                    for i in range(frame_input_len):
                        if int(index+consecutive_frames) >= frame_input_len:
                            break
                        temporal_input = np.dstack(frame_input['hori'][index:index+consecutive_frames] +
                                                   frame_input['vert'][index:index+consecutive_frames])
                        spatial_input = frame_input['orig'][int(index+consecutive_frames//2)]
                        index += consecutive_frames

                        temporal_input_list.append(temporal_input)
                        spatial_input_list.append(spatial_input)
                    return temporal_input_list, spatial_input_list
                # ------------------------------------------------

                # training set or testing set
                # --------------------------------------------------------------------------
                if file_name in training_index['name']:
                    temporal_input_list, spatial_input_list = allocate_input_data()
                    for i in range(np.min([len(temporal_input_list), np.min(len(spatial_input_list))])):
                        training_set['input']['temporal'].append(temporal_input_list[i])
                        training_set['input']['spatial'].append(spatial_input_list[i])
                        training_set['label'].append(training_index['label'][training_index['name'].index(file_name)])
                elif file_name in testing_index['name']:
                    temporal_input_list, spatial_input_list = allocate_input_data()
                    for i in range(np.min([len(temporal_input_list), np.min(len(spatial_input_list))])):
                        testing_set['input']['temporal'].append(temporal_input_list[i])
                        testing_set['input']['spatial'].append(spatial_input_list[i])
                else:
                    print("video: '%s' isn't used for training or testing." % file_name)
                    continue
                # print('%s .. ok' % file_name)
                # --------------------------------------------------------------------------

                # only for testing program
                # --------------------------------------
                # if len(training_set['input']['temporal']) > test_program_flag:
                #     print('just for testing program')
                #     break
                # --------------------------------------

        elif os.path.isfile(file_directory):
            print('not a directory')
            exit()
    else:
        print("The specified directory isn't exist.")
        exit()
    # --------------------------------------------------

    # save pickle
    # ----------------------------------------------------
    with open(pickle_directory+'train.pickle', 'wb') as fw:
        cPickle.dump(training_set, fw)
    with open(pickle_directory + 'test.pickle', 'wb') as fw:
        cPickle.dump(testing_set, fw)
    print('Pickle files have been saved.')
    # ----------------------------------------------------
    del training_index, testing_index, training_files, testing_files

    return training_set, testing_set


def get_data_set():
    print('Get data set.')
    training_set, testing_set = stack_optical_flow(file_directory)
    return training_set, testing_set


if __name__ == "__main__":
    # create_optical_flow(file_directory+file_name)
    stack_optical_flow(file_directory)
