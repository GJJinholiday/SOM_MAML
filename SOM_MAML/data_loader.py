import numpy as np
import pandas as pd
import os

#In this module, we build the dataset to train the traffic data embedding which is used to classify by SOM.

class loader:
    #Set up some parameter.
    def __init__(self, type='pretrain-NYtaxi'):
        self.timeslot_daynum = 48
        self.threshold = 0.05
        self.pretrain_NYtaxi_path = './Data/Pretrain_data_NYtaxi/NYtaxi_pretrain.npy'
        self.finetune_SZtaxi_path = './Data/Finetune_data_SZtaxi/SZtaxi_finetune.npy'
        self.finetune_NYbike_path = './Data/Finetune_data_NYbike/NYbike_finetune.npy'
        self.test_SZtaxi_path = './Data/Finetune_data_SZtaxi/SZtaxi_test.npy'
        self.test_NYbike_path = './Data/Finetune_data_NYbike/NYbike_test.npy'

        self.type = type

    #This function load the dataset.
    def load_volume(self):
        if self.type == 'pretrain-NYtaxi':
            print('loading pretrain-NYtaxi......')
            self.data = np.load(self.pretrain_NYtaxi_path)
            self.max = np.max(self.data)
            self.data = self.data/self.max
        elif self.type == 'finetune-SZtaxi':
            print('loading finetune-SZtaxi......')
            self.data = np.load(self.finetune_SZtaxi_path)
            self.max = np.max(self.data)
            self.data = self.data / self.max
        elif self.type == 'finetune-NYbike':
            print('loading finetune-NYbike......')
            self.data = np.load(self.finetune_NYbike_path)
            self.max = np.max(self.data)
            self.data = self.data / self.max
        elif self.type == 'test-SZtaxi':
            print('loading test-SZtaxi......')
            self.data = np.load(self.test_SZtaxi_path)
            self.max = np.max(self.data)
            self.data = self.data / self.max
        elif self.type == 'test-NYbike':
            print('loading test-NYbike......')
            self.data = np.load(self.test_NYbike_path)
            self.max = np.max(self.data)
            self.data = self.data / self.max

    #This function features for cnn and lstm.
    def sample_embedding(self, type, short_term_lstm_seq_len=7, cnn_nbhd_size=1):
        self.type = type
        self.load_volume()
        cnn_features = []
        for i in range(short_term_lstm_seq_len):
            cnn_features.append([])
        names = []
        labels = []

        time_start = short_term_lstm_seq_len
        #time_end = time_start
        time_end = self.data.shape[0]
        volume_type = self.data.shape[-1]

        for t in range(time_start, time_end):
            if t % 100 == 0:
                print("Now sampling at {0} timeslots.".format(t))
            for x in range(self.data.shape[1]):
                for y in range(self.data.shape[2]):

                    # label
                    if (self.data[t, x, y, 0] > self.threshold and self.data[t, x, y, 1] > self.threshold and self.data[t, x, y, 0] < 1 and self.data[t, x, y, 1] < 1):
                        labels.append(self.data[t, x, y, :].flatten())
                    else:
                        continue

                    # sample common (short-term) lstm
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        # cnn features, zero_padding
                        cnn_feature = np.zeros((2 * cnn_nbhd_size + 1, 2 * cnn_nbhd_size + 1, volume_type))
                        # actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                # boundary check
                                if not (0 <= cnn_nbhd_x < self.data.shape[1] and 0 <= cnn_nbhd_y < self.data.shape[2]):
                                    continue
                                # get features
                                cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size),
                                :] = self.data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        cnn_features[seqn].append(cnn_feature)

                    # name
                    names.append(str(t)+":("+str(x)+","+str(y)+")")



        for i in range(short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])

        labels = np.array(labels)

        return cnn_features, names, labels