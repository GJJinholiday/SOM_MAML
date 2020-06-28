import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.models import Model
import keras
import numpy as np
import data_loader
import embedding_model
from keras.callbacks import EarlyStopping
import datetime
import argparse
import gc

def classify(short_term_lstm_seq_len, number_class, type, res, savepath):
    sampler = data_loader.loader()
    x, names, y = sampler.sample_embedding(type=type, short_term_lstm_seq_len=short_term_lstm_seq_len)
    x=np.array(x)
    y=np.array(y)
    res=np.squeeze(res)
    print(res.shape)
    print(x.shape)
    print(y.shape)
    print(set(res.tolist()))
    for i in np.arange(number_class):
        print(i)
        index = np.array(np.squeeze(np.argwhere(res==i)))
        iclass_data = np.array([x[:,index,:,:,:], y[index, :]])
        print(np.array(iclass_data[0]).shape)
        print(np.array(iclass_data[1]).shape)
        np.save(savepath+'class_data/'+str(int(i))+'class', iclass_data)
        del iclass_data
        gc.collect()


classify(7, 20, type="finetune-SZtaxi", res=np.load('./Data/Finetune_data_SZtaxi/SZtaxi_finetune_res.npy'), savepath='./Data/Finetune_data_SZtaxi/finetune_')
classify(7, 20, type="finetune-NYbike", res=np.load('./Data/Finetune_data_NYbike/NYbike_finetune_res.npy'), savepath='./Data/Finetune_data_NYbike/finetune_')
classify(7, 20, type="test-NYbike", res=np.load('./Data/Finetune_data_NYbike/NYbike_test_res.npy'), savepath='./Data/Finetune_data_NYbike/test_')
