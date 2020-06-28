import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras.models import Model
import keras
import numpy as np
import data_loader
from keras.models import load_model

model = load_model('embedding_model')

layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding').output)

sampler = data_loader.loader()

for i, j in zip(['./Data/Finetune_data_SZtaxi/SZtaxi_finetune', './Data/Finetune_data_NYbike/NYbike_finetune', './Data/Finetune_data_SZtaxi/SZtaxi_test', './Data/Finetune_data_NYbike/NYbike_test'], ['finetune-SZtaxi', 'finetune-NYbike', 'test-SZtaxi', 'test-NYbike']):
    x, names, y = sampler.sample_embedding(type=j, short_term_lstm_seq_len=7)


    feature = np.array(layer_model.predict(x=x,))

    print(feature.shape)

    np.save(i+'_embedding', feature)
    print(j)
    print(np.array(x).shape)
    print(np.array(feature).shape)