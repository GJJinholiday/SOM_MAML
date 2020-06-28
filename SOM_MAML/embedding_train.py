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

parser = argparse.ArgumentParser(description='Data embedding')
parser.add_argument('--type', type=str, default='pretrain-NYtaxi')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of batch')
parser.add_argument('--max_epochs', type=int, default=1000,
                    help='maximum epochs')
parser.add_argument('--short_term_lstm_seq_len', type=int, default=7,
                    help='the length of short term value')
parser.add_argument('--model_name', type=str, default='embedding_model',
                    help='model name')

args = parser.parse_args()
print(args)


class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def eval_together(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask) == 0:
        return -1
    mape = np.mean(np.abs(y[mask] - pred_y[mask]) / y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))

    return rmse, mape


def eval_lstm(y, pred_y, threshold):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    # pickup part
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]) / pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask])))
    # dropoff part
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(
            np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]) / dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask])))

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)


def main(batch_size=64, max_epochs=100, validation_split=0.2, early_stop=EarlyStopping()):
    model_hdf5_path = "./hdf5s/"

    sampler = data_loader.loader()

    modeler = embedding_model.embedding_model()

    if args.model_name == "embedding_model":
        # training
        x, names, y = sampler.sample_embedding(type="pretrain-NYtaxi", short_term_lstm_seq_len=args.short_term_lstm_seq_len)


        #print(np.array(x).shape)
        #print(np.array(names).shape)
        #print(np.array(y).shape)

        model = modeler.embedding(embedding_shape=128, nbhd_size=1, volume_type=2, cnn_flat_size=128,lstm_seq_len=args.short_term_lstm_seq_len)

        model.fit(
            x=x,
            y=y,
            batch_size=batch_size, validation_split=validation_split, epochs=max_epochs, callbacks=[early_stop])

        model.save('embedding_model')

        layer_model = Model(inputs=model.input, outputs=model.get_layer('embedding').output)

        feature = np.array(layer_model.predict(x=x,))

        np.save('./Data/Pretrain_data_NYtaxi/NYtaxi_pretrain_embedding', feature)

        return

    else:
        print("Cannot recognize parameter...")
        return

if __name__ == '__main__':
    stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=40)
    main(batch_size=args.batch_size, max_epochs=args.max_epochs, early_stop=stop)