# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import os
import cv2
from config import root

root_path = root()
file_directory = root_path + 'dataset/UCF101/'
# file_name = 'v_Archery_g01_c07.avi'
index_directory = root_path + 'dataset/ucfTrainTestlist/'


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)  # 以網格的形式選取二维圖像上等間隔的點，這裡間隔为16，reshape成2行的array
    fx, fy = flow[y, x].T  # 取選定網格點座標對應的光流位移
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)  # 将初始點和變化的點堆疊成2*2的數組
    lines = np.int32(lines + 0.5)  # 忽略微笑的假偏移，整數化
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))  # 以初始點和終點劃線表示光流運動
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)  # 在初始點（網格點處畫圆點来表示初始點）
    return vis


def create_optical_flow(video_file, sample_freq_of_motion=1):
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
        frame_set['orig'].append(frame)
        frame_set['gray'].append(cv2.cvtColor(frame_set['orig'][count], cv2.COLOR_BGR2GRAY))

        # calculate motion of two frames
        if ((count % sample_freq_of_motion) == 0) and (count != 0):
            frame_set['flow'].append(cv2.calcOpticalFlowFarneback(frame_set['gray'][count-sample_freq_of_motion],
                                                                  frame_set['gray'][count],
                                                                  None, 0.5, 3, 15, 3, 5, 1.1, 0))
            # Display flow
            # cv2.imshow('frame', draw_flow(frame_set['gray'][count], frame_set['flow'][-1]))
            # cv2.waitKey(100)

            # horizontal & vertical
            frame_set['hori'].append(cv2.normalize(frame_set['flow'][-1][..., 0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))
            frame_set['vert'].append(cv2.normalize(frame_set['flow'][-1][..., 1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))

            # Display flow
            # cv2.imshow('frame', frame_set['vert'][-1])
            # cv2.waitKey(100)
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

    # make a list to record all video in this directory
    # -------------------------------------------------
    if os.path.exists(file_directory):
        if os.path.isdir(file_directory):
            file_list = os.listdir(file_directory)
            for file_name in file_list:
                # Allocate input data to temporal and spatial set
                # ------------------------------------------------
                frame_input = create_optical_flow(file_directory + file_name)
                temporal_input = np.dstack(frame_input['hori'] + frame_input['vert'])
                spatial_input = frame_input['orig']
                # ------------------------------------------------

                if file_name in training_index['name']:
                    training_set['input']['temporal'].append(temporal_input)
                    training_set['input']['spatial'].append(spatial_input)
                    training_set['label'].append(training_index['label'][training_index['name'].index(file_name)])
                elif file_name in testing_index['name']:
                    testing_set['input']['temporal'].append(temporal_input)
                    testing_set['input']['spatial'].append(spatial_input)
                else:
                    print("video: '%s' isn't used for training or testing." % file_name)
                    continue
                print('%s .. ok' % file_name)

        elif os.path.isfile(file_directory):
            exit()
    else:
        print("The specified directory isn't exist.")
        exit()
    # --------------------------------------------------

    return 0


if __name__ == "__main__":
    # create_optical_flow(file_directory+file_name)
    stack_optical_flow(file_directory)
