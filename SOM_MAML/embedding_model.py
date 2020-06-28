from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input, LSTM, Conv2D, Flatten, Concatenate, Reshape
import numpy as np

class embedding_model:
    def __init__(self):
        pass#self.embedding_record = {}


    def embedding(self, embedding_shape, nbhd_size=1, volume_type=2, lstm_seq_len=4, cnn_flat_size=128, optimizer = 'adagrad', loss = 'mse', metrics=[]):
        nbhd_inputs = [Input(shape=(2*nbhd_size+1, 2*nbhd_size+1, volume_type,), name="nbhd_volume_input_time_{0}".format(ts+1)) for ts in range(lstm_seq_len)]
        #name = Input(shape=(1,), name="label")

        # lstm_cnn
        nbhd_convs = [Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="nbhd_convs_time0_{0}".format(ts+1))(nbhd_inputs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time0_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="nbhd_convs_time1_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time1_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Conv2D(filters=128, kernel_size=(3, 3), padding="same", name="nbhd_convs_time2_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time2_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]

        nbhd_vecs = [Flatten(name="nbhd_flatten_time_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(lstm_seq_len)]
        nbhd_vecs = [Dense(units=cnn_flat_size, name="nbhd_dense_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for ts in
                     range(lstm_seq_len)]
        nbhd_vecs = [Activation("relu", name="nbhd_dense_activation_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for ts in
                     range(lstm_seq_len)]
        nbhd_vecs = Concatenate(axis=-1)(nbhd_vecs)
        lstm_inputs = Reshape(target_shape=(lstm_seq_len, cnn_flat_size))(nbhd_vecs)

        # lstm
        lstm = LSTM(units=cnn_flat_size, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(lstm_inputs)
        # lstm_all = Dropout(rate = .3)(lstm_all)
        lstm = Dense(units=embedding_shape, name='embedding')(lstm)

        #self.embedding_record[name] = lstm

        lstm = Dense(units=volume_type)(lstm)

        pred_volume = Activation('tanh')(lstm)

        inputs = nbhd_inputs
        # print("Model input length: {0}".format(len(inputs)))
        # ipdb.set_trace()
        model = Model(inputs=inputs, outputs=pred_volume)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model