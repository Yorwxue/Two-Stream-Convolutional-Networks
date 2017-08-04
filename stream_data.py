# -*- coding: utf-8 -*-
from __future__ import print_function

import cPickle as pickle
import numpy as np
import os
import cv2
import tqdm
import random
# import time
import gc

from config import get_parameter, time_spent_printer

parameter = get_parameter()
root_path = parameter['root_path']
consecutive_frames = parameter['consecutive_frames']
img_rows = parameter['img_cols']
img_cols = parameter['img_cols']

file_directory = parameter['file_directory']
pickle_directory = parameter['pickle_directory']
index_directory = parameter['index_directory']

sample_freq_of_motion = parameter['sample_freq_of_motion']
split_selection = parameter['split_selection']

# file_name = 'v_Archery_g01_c07.avi'
# file_name = 'v_Skiing_g21_c01.avi'

data_update = False  # True / False

test_program_flag = 10  # only for testing program


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


def create_optical_flow(video_file, sample_freq_of_motion):
    # video_file: the directory and file name of video
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


def stack_optical_flow(file_directory, data_update=False):
    # create dictionary of classes and indices
    # ------------------------------------------------
    with open(index_directory + 'classInd.txt', 'r') as fr:
        lines = fr.readlines()
        class_index = [entry.replace('\r\n', '').split(' ') for entry in lines]

    class_index_dict = dict()
    classes_of_videos_dict = dict()
    for i in range(len(class_index)):
        classes_of_videos_dict[class_index[i][1]] = list()
        class_index_dict[class_index[i][0]] = class_index[i][1]
        class_index_dict[class_index[i][1]] = class_index[i][0]

    # save pickle
    with open(pickle_directory+'class_index_dict.pickle', 'wb') as fw:
        pickle.dump(class_index_dict, fw, protocol=pickle.HIGHEST_PROTOCOL)
    # ------------------------------------------------

    # make a list to record all videos in this directory
    # -------------------------------------------------------------------
    if os.path.exists(file_directory):
        if os.path.isdir(file_directory):
            file_list = os.listdir(file_directory)
            pickle_list = os.listdir(pickle_directory)  # videos which already dump into puckle
            num_of_files = len(file_list)

            # for file_name in file_list:
            for file_index in tqdm.tqdm(range(num_of_files)):  # test_program_flag
                file_name = file_list[file_index]
                class_name = file_name.split('_')[1]

                if file_name not in classes_of_videos_dict[class_name]:
                    classes_of_videos_dict[class_name].append(file_name)

                if not data_update and (file_name + '.pickle') in pickle_list:
                    continue

                video = dict()
                video['label'] = class_index_dict[class_name]

                # Allocate input data to temporal and spatial set
                # ------------------------------------------------
                def allocate_input_data(video_file):
                    index = 0
                    temporal_input_list = list()
                    spatial_input_list = list()

                    frame_input = create_optical_flow(video_file, sample_freq_of_motion)
                    frame_input_len = np.min([len(frame_input['hori']), len(frame_input['vert'])])  # length should be the same
                    for i in range(frame_input_len):
                        if int(index + consecutive_frames) >= frame_input_len:
                            break
                        temporal_input = np.dstack(frame_input['hori'][index:index + consecutive_frames] +
                                                   frame_input['vert'][index:index + consecutive_frames])
                        spatial_input = frame_input['orig'][int(index + consecutive_frames // 2)]
                        index += consecutive_frames

                        temporal_input_list.append(temporal_input)
                        spatial_input_list.append(spatial_input)
                    return temporal_input_list, spatial_input_list
                # ------------------------------------------------

                temporal_input_list, spatial_input_list = allocate_input_data(file_directory + file_name)
                video['input'] = dict()
                video['input']['temporal'] = list()
                video['input']['spatial'] = list()

                for i in range(np.min([len(temporal_input_list), len(spatial_input_list)])):  # two list should have the same length
                    video['input']['temporal'].append(temporal_input_list[i])
                    video['input']['spatial'].append(spatial_input_list[i])

                # save pickle
                # ------------------------------------------------
                with open(pickle_directory + '%s.pickle' % file_name, 'wb') as fw:
                    pickle.dump(video, fw, protocol=pickle.HIGHEST_PROTOCOL)
                # ------------------------------------------------

            with open(pickle_directory + 'classes_of_videos_dict.pickle', 'wb') as fw:
                pickle.dump(classes_of_videos_dict, fw, protocol=pickle.HIGHEST_PROTOCOL)

            print('Videos processed')

        elif os.path.isfile(file_directory):
            print('not a directory')
            exit()
    else:
        print("The specified directory isn't exist.")
        exit()
    # -------------------------------------------------------------------


def get_data_set(class_index_dict, seed, sample_times, mini_batch=256, kind='train', data_update=False):
    # seed: shuffle seed
    # sample_times: which times of sampling
    # mini_batch: return how many data
    # kind: train/test
    data_set = dict()
    data_set['input'] = dict()
    data_set['input']['temporal'] = list()
    data_set['input']['spatial'] = list()
    data_set['label'] = list()

    while True:
        if kind != 'train' and kind != 'test':
            print("kind must be 'train' or 'test'.")
            kind = input('type in again.(train/test)')
        else:
            break

    # data split of training/testing
    # ----------------------------------------------------------------------
    # data_index: all names of videos in the training/testing set, ex: 'v_Archery_g01_c07.avi'
    # if kind == 'train':
    #     print('Prepare the data set of %dth round.' % (sample_times+1))
    data_index = dict()
    data_index['name'] = list()
    # data_index['label'] = list()

    data_files_list = ['%slist01.txt' % kind, '%slist02.txt' % kind, '%slist03.txt' % kind]

    for data_file in data_files_list:
        with open(index_directory + '%s' % data_file, 'r') as fr:
            lines = fr.readlines()
            data_index['name'] += [entry.split(' ')[0][entry.split(' ')[0].index('/') + 1:].replace('\r\n', '') for entry in lines]
            # data_index['label'] += [entry.split(' ')[1].replace('\r\n', '') for entry in lines]
    # ----------------------------------------------------------------------

    # data shuffle
    # seed = [random.random() for i in range(num_of_classes)]

    # data collector
    num_of_classes = len(class_index_dict) / 2
    num_of_samples_of_each_class = mini_batch // num_of_classes
    num_of_samples_of_remaining = mini_batch % num_of_classes
    samples_of_remaining = np.random.randint(0, num_of_classes, num_of_samples_of_remaining)

    # start_time = time.time()
    for i in tqdm.tqdm(range(num_of_classes)):
        # Reading pickle
        # -------------------------------------------------------------------------------
        with open(pickle_directory+'classes_of_videos_dict.pickle', 'rb') as fr:
            classes_of_videos_dict = pickle.load(fr)

        with open(pickle_directory + 'class_index_dict.pickle', 'rb') as fr:
            class_index_dict = pickle.load(fr)
        # -------------------------------------------------------------------------------

        # Select videos
        # ---------------------------------------------------------------------------
        # sample_list = np.arange(len(data_index['name']))[0:mini_batch]
        # np.random.shuffle(sample_list)
        #
        video_list_of_same_class = classes_of_videos_dict[class_index_dict[str(i+1)]]
        video_list_of_same_class = sorted(video_list_of_same_class)
        random.shuffle(video_list_of_same_class, lambda: seed[i])
        # ---------------------------------------------------------------------------

        # Samples from each class
        # --------------------------------------------------------------------------
        for j in range(num_of_samples_of_each_class):
            # Select one video
            # ----------------------------------------------------------------------
            video_selection = j + sample_times * num_of_samples_of_each_class
            video_name = video_list_of_same_class[video_selection]

            with open(pickle_directory + '%s.pickle' % video_name, 'rb') as fr:
                video = pickle.load(fr)
            # ----------------------------------------------------------------------

            # Collect data
            # ----------------------------------------------------------------------
            temporal_data_set = video['input']['temporal']
            spatial_data_set = video['input']['spatial']
            data_label = int(class_index_dict[
                                 class_index_dict[str(i+1)]  # class name
                             ]) - 1  # start from 0
            # ----------------------------------------------------------------------

            # Randomly select one data from this video
            # ----------------------------------------------------------------------
            try:
                selection = np.random.randint(0, 999) % len(temporal_data_set)

                data_set['input']['temporal'].append(temporal_data_set[selection])
                data_set['input']['spatial'].append(spatial_data_set[selection])
                data_set['label'].append(data_label)
            except:
                None
            # ----------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # Random sample
        # -------------------------------------------
        if i in samples_of_remaining:
            # Randomly select one video
            # ----------------------------------------------------------------------
            random_select = np.random.randint(0, len(video_list_of_same_class))
            video_name = video_list_of_same_class[random_select]
            with open(pickle_directory + '%s.pickle' % video_name, 'rb') as fr:
                video = pickle.load(fr)
            # ----------------------------------------------------------------------

            # Collect data
            # ----------------------------------------------------------------------
            temporal_data_set = video['input']['temporal']
            spatial_data_set = video['input']['spatial']
            data_label = int(class_index_dict[
                                 class_index_dict[str(i + 1)]  # class name
                             ]) - 1  # start from 0
            # ----------------------------------------------------------------------

            # Randomly select one data from this video
            # ----------------------------------------------------------------------
            try:
                selection = np.random.randint(0, 999) % len(temporal_data_set)

                data_set['input']['temporal'].append(temporal_data_set[selection])
                data_set['input']['spatial'].append(spatial_data_set[selection])
                data_set['label'].append(data_label)
            except:
                None
            # ----------------------------------------------------------------------
        # -------------------------------------------
        gc.collect()
        # only for test
        # if i > test_program_flag:
        #     break

    # end_time = time.time()
    # time_spent_printer(start_time, end_time)

    return data_set


class data_set:
    # seed: shuffle seed
    # sample_times: which times of sampling
    # mini_batch: return how many data
    # kind: train/test
    data_set = dict()

    def __init__(self, class_index_dict, kind='train'):
        while True:
            if kind != 'train' and kind != 'test':
                print("kind must be 'train' or 'test'.")
                kind = input('type in again.(train/test)')
            else:
                break

        # data split of training/testing
        # ----------------------------------------------------------------------
        # data_index: all names of videos in the training/testing set, ex: 'v_Archery_g01_c07.avi'
        data_index = dict()
        data_index['name'] = list()

        data_files_list = ['%slist%s.txt' % (kind, split_selection)]

        for data_file in data_files_list:
            with open(index_directory + '%s' % data_file, 'r') as fr:
                lines = fr.readlines()
                data_index['name'] += [entry.split(' ')[0][entry.split(' ')[0].index('/') + 1:].replace('\r\n', '') for entry in lines]

                # a bug of data set split provided by UCF101
                # ----------------------------------------------------------------------
                for check_name in range(len(data_index['name'])):
                    if 'HandStandPushups' in data_index['name'][check_name]:
                        data_index['name'][check_name] = data_index['name'][check_name].replace('HandStandPushups', 'HandstandPushups')
                # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Reading pickle
        # -------------------------------------------------------------------------------
        with open(pickle_directory + 'classes_of_videos_dict.pickle', 'rb') as fr:
            self.classes_of_videos_dict = pickle.load(fr)

        with open(pickle_directory + 'class_index_dict.pickle', 'rb') as fr:
            self.class_index_dict = pickle.load(fr)
            # -------------------------------------------------------------------------------

        # start_time = time.time()
        for i in tqdm.tqdm(range(len(data_index['name']))):
            # read videos
            # --------------------------
            video_name = data_index['name'][i]
            class_name = video_name.split('_')[1]
            # try:
            with open(pickle_directory + '%s.pickle' % video_name, 'rb') as fr:
                video = pickle.load(fr)
            # except:
            #     continue
            # --------------------------

            # Collect data
            # ----------------------------------------------------------------------
            temporal_data_set = video['input']['temporal']
            spatial_data_set = video['input']['spatial']
            data_label = int(self.class_index_dict[class_name]) - 1  # start from 0
            # ----------------------------------------------------------------------

            # create data set
            # -------------------------------------
            if class_name not in self.data_set:
                self.data_set[class_name] = dict()
            if 'input' not in self.data_set[class_name]:
                self.data_set[class_name]['input'] = dict()
            if 'temporal' not in self.data_set[class_name]['input']:
                self.data_set[class_name]['input']['temporal'] = list()
            if 'spatial' not in self.data_set[class_name]['input']:
                self.data_set[class_name]['input']['spatial'] = list()
            if 'label' not in self.data_set[class_name]:
                self.data_set[class_name]['label'] = data_label
            # -------------------------------------

            # Randomly select one data from this video
            # ----------------------------------------------------------------------
            self.data_set[class_name]['input']['temporal'].append(temporal_data_set)
            self.data_set[class_name]['input']['spatial'].append(spatial_data_set)
            # ----------------------------------------------------------------------
            # only for test
            # if i > 500:
            #     break
        # only for test
        # print(len(self.data_set.keys()))
        # print(self.data_set.keys())
        gc.collect()

    def get_minibatch(self, seed, sample_times, mini_batch=256):
        # data collector
        num_of_classes = len(self.class_index_dict) / 2
        num_of_samples_of_each_class = mini_batch // num_of_classes
        num_of_samples_of_remaining = mini_batch % num_of_classes
        samples_of_remaining = np.random.randint(0, num_of_classes, num_of_samples_of_remaining)
        samples_of_remaining = samples_of_remaining.tolist()

        get_data = dict()
        get_data['input'] = dict()
        get_data['input']['temporal'] = list()
        get_data['input']['spatial'] = list()
        get_data['label'] = list()

        for i in tqdm.tqdm(range(num_of_classes)):
            class_name = self.class_index_dict[str(i + 1)]

            # Select videos
            # ---------------------------------------------------------------------------
            sample_list_of_this_class = np.arange(len(self.data_set[class_name]['input']['temporal']))
            # np.random.shuffle(sample_list)
            #
            # video_list_of_same_class = self.classes_of_videos_dict[class_index_dict[str(i + 1)]]
            # video_list_of_same_class = sorted(video_list_of_same_class)
            random.shuffle(sample_list_of_this_class, lambda: seed[i])
            # ---------------------------------------------------------------------------

            # Samples from each class
            # --------------------------------------------------------------------------
            for j in range(num_of_samples_of_each_class):
                # Select one video
                # ----------------------------------------------------------------------
                video_selection = j + sample_times * num_of_samples_of_each_class
                if video_selection >= len(sample_list_of_this_class):
                    video_selection = np.random.randint(0, len(sample_list_of_this_class))
                else:
                    video_selection = sample_list_of_this_class[video_selection]

                # video_name = video_list_of_same_class[video_selection]
                #
                # with open(pickle_directory + '%s.pickle' % video_name, 'rb') as fr:
                #     video = pickle.load(fr)
                # ----------------------------------------------------------------------

                try:
                    # Collect data
                    # ----------------------------------------------------------------------
                    temporal_data_set = self.data_set[class_name]['input']['temporal'][video_selection]
                    spatial_data_set = self.data_set[class_name]['input']['spatial'][video_selection]
                    data_label = self.data_set[class_name]['label']
                    # ----------------------------------------------------------------------

                    # Randomly select one data from this video
                    # ----------------------------------------------------------------------
                    selection = np.random.randint(0, 999) % len(temporal_data_set)

                    get_data['input']['temporal'].append(temporal_data_set[selection])
                    get_data['input']['spatial'].append(spatial_data_set[selection])
                    get_data['label'].append(data_label)
                    # ----------------------------------------------------------------------
                except:
                    # print('class name: %s' % class_name)
                    # print('selection of data of this video: %d' % video_selection)
                    # print('number of data of this video: %d' % len(temporal_data_set))
                    # input('push any key to continue.')
                    None
            # --------------------------------------------------------------------------

            # Random sample
            # -------------------------------------------
            while i in samples_of_remaining:
                samples_of_remaining.pop(samples_of_remaining.index(i))

                # Randomly select one video
                # ----------------------------------------------------------------------
                random_selection = np.random.randint(0, len(sample_list_of_this_class))
                # video_name = video_list_of_same_class[random_select]
                # with open(pickle_directory + '%s.pickle' % video_name, 'rb') as fr:
                #     video = pickle.load(fr)
                # ----------------------------------------------------------------------

                try:
                    # Collect data
                    # ----------------------------------------------------------------------
                    temporal_data_set = self.data_set[class_name]['input']['temporal'][random_selection]
                    spatial_data_set = self.data_set[class_name]['input']['spatial'][random_selection]
                    data_label = int(self.data_set[class_name]['label']) - 1  # start from 0
                    # ----------------------------------------------------------------------

                    # Randomly select one data from this video
                    # ----------------------------------------------------------------------
                    selection = np.random.randint(0, 999) % len(temporal_data_set)

                    get_data['input']['temporal'].append(temporal_data_set[selection])
                    get_data['input']['spatial'].append(spatial_data_set[selection])
                    get_data['label'].append(data_label)
                    # ----------------------------------------------------------------------
                except:
                    # print('class name: %s' % class_name)
                    # print('selection of data of this video: %d' % video_selection)
                    # print('number of data of this video: %d' % len(temporal_data_set))
                    # input('push any key to continue.')
                    None
            # -------------------------------------------
        gc.collect()
        return get_data


if __name__ == "__main__":
    # create_optical_flow(file_directory+file_name)


    # stack_optical_flow(file_directory, data_update=data_update)  #create pickle for training and testing


    with open(pickle_directory + 'class_index_dict.pickle', 'rb') as fr:
        class_index_dict = pickle.load(fr)
    num_of_classes = len(class_index_dict) / 2
    seed = [random.random() for i in range(num_of_classes)]

    # ver 1 pickle of video
    # get_data_set(class_index_dict, seed, 0)

    # ver 2 pickle of class
    # get_data_set_ver2(class_index_dict)

    # ver 3 server version
    a = data_set(class_index_dict, kind='train')
    a.get_minibatch(seed, 0)
