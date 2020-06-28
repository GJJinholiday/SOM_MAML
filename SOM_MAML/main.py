import os
import numpy as np
import argparse
import tensorflow as tf

from data_generater import DataGenerator_SOM_MAML_with_attention
from maml import MAML, NO_MAML

tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='pretrain-NYbike', help='set for pretrain-NYbike, pretrain-SZtaxi, train_without_pretrain-NYbike or train_without_pretrain-SZtaxi')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def pretrain(model, saver, sess, iteartions):
    """

    :param model:
    :param saver:
    :param sess:
    :return:
    """
    print('-----------------------------------------------')
    print('||                pretrain                   ||')
    print('-----------------------------------------------')
    # write graph to tensorboard
    # tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)
    prelosses, postlosses = [], []
    premapes, postmapes = [], []

    # train for meta_iteartion epoches
    for iteration in range(int(iteartions*(9/10))):
        # this is the main op
        ops = [model.meta_op]

        # add summary and print op
        if iteration % 100 == 0:
            print('The ', iteration, ' ing')
            ops.extend([model.query_losses[0], model.query_losses[-1], model.query_mapes[0], model.query_mapes[-1]])

        # run all ops
        result = sess.run(ops)

        # summary
        if iteration % 100 == 0:
            # summ_op
            # tb.add_summary(result[1], iteration)
            # query_losses[0]
            prelosses.append(result[1])
            postlosses.append(result[2])
            premapes.append(result[3])
            postmapes.append(result[4])
            print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses), '       ',  '\tmape:', np.mean(premapes), '=>', np.mean(postmapes))
            prelosses, postlosses= [], []
            premapes, postmapes = [], []

        # evaluation
        #if iteration % 100 == 0:
        #    # DO NOT write as a = b = [], in that case a=b
        #    # DO NOT use train variable as we have train func already.
        #    loss = np.mean(postlosses)
        #    saver.save(sess, os.path.join('ckpt/pretrain', 'mini.mdl'))
        #    minloss = loss
        #    print('saved into ckpt:', minloss)

def fine_tune(model, saver, sess, iterations):
    """

        :param model:
        :param saver:
        :param sess:
        :return:
        """
    # write graph to tensorboard
    # tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)

    print('-----------------------------------------------')
    print('||                fine-tune                  ||')
    print('-----------------------------------------------')
    finetunelosses = []
    finetunemapes = []
    min = 100

    # train for meta_iteartion epoches
    for iteration in range(iterations):
        # this is the main op
        ops = [model.finetune_op, model.mape]

        # add summary and print op
        if iteration % 100 == 0:
            print('The ', iteration, ' ing')
            ops.extend([model.loss])

        # run all ops
        result = sess.run(ops)
        if np.mean(result[1]) < min:
            min = np.mean(result[1])

        # summary
        if iteration % 100 == 0:
            # summ_op
            # tb.add_summary(result[1], iteration)
            # query_losses[0]
            finetunelosses.append(result[2])
            finetunemapes.append(result[1])

            # query_losses[-1]
            print(iteration, '\ttestloss:', np.mean(finetunelosses), '      ','\ttestmape:', np.mean(finetunemapes))
            print("the min mape:" + str(min))
            finetunelosses = []
            finetunemapes = []

        # save session
        #if iteration % 100 == 0:
        #    # DO NOT write as a = b = [], in that case a=b
        #    # DO NOT use train variable as we have train func already.
        #    loss = np.mean(testlosses)
        #    saver.save(sess, os.path.join('ckpt/pretrain', 'mini.mdl'))
        #    print('saved into ckpt:', loss)
    print("the min mape:" + str(min))

def train_without_pretrain(model, saver, sess, iterations):
    """

            :param model:
            :param saver:
            :param sess:
            :return:
            """
    # write graph to tensorboard
    # tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)
    print('-----------------------------------------------')
    print('||           train_without_pretrain          ||')
    print('-----------------------------------------------')
    losses = []
    mapes = []
    min = 100

    # train for meta_iteartion epoches
    for iteration in range(iterations):
        # this is the main op
        ops = [model.train_op, model.mape]

        # add summary and print op
        if iteration % 100 == 0:
            print('The ', iteration, ' ing')
            ops.extend([model.loss])

        # run all ops
        result = sess.run(ops)
        if np.mean(result[1]) < min:
            min = np.mean(result[1])

        # summary
        if iteration % 100 == 0:
            # summ_op
            # tb.add_summary(result[1], iteration)
            # query_losses[0]
            losses.append(result[2])
            mapes.append(result[1])
            print(iteration, '\tloss:', np.mean(losses), '     ',  '\tmape:', np.mean(mapes))
            print("the min mape:" + str(min))
            losses = []
            mapes = []
    print("the min mape:" + str(min))

        # save session
        #if iteration % 100 == 0:
        #    # DO NOT write as a = b = [], in that case a=b
        #    # DO NOT use train variable as we have train func already.
        #    loss = np.mean(testlosses)
        #    saver.save(sess, os.path.join('ckpt/train_without_pretrain', 'mini.mdl'))
        #    print('saved into ckpt:', loss)

def shenzhen_SZ_test(short_term_lstm_seq_len=7, cnn_nbhd_size=1, threshold=0.05):
    data = np.load('./Data/Finetune_data_SZtaxi/SZtaxi_test.npy', allow_pickle=True)
    labeltotal = np.load('./Data/Finetune_data_SZtaxi/SZtaxi_test_label.npy', allow_pickle=True)
    print(data.shape)
    print(labeltotal.shape)
    max = np.max(data)
    data = data / max
    cnn_features = []
    for i in range(short_term_lstm_seq_len):
        cnn_features.append([])
    labels = []

    time_start = short_term_lstm_seq_len
    # time_end = time_start
    time_end = data.shape[0]
    volume_type = data.shape[-1]
    print(data.shape)

    for t in range(time_start, time_end):
        if (labeltotal[t])==(labeltotal[t-1]+1) and (labeltotal[t])==(labeltotal[t-2]+2) and (labeltotal[t])==(labeltotal[t-3]+3) and (labeltotal[t])==(labeltotal[t-4]+4) and (labeltotal[t])==(labeltotal[t-5]+5) and (labeltotal[t])==(labeltotal[t-6]+6) and (labeltotal[t])==(labeltotal[t-7]+7):
            if t % 100 == 0:
                print("Now sampling at {0} timeslots.".format(t))
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):

                    # label
                    if (data[t, x, y, 0] > threshold and data[t, x, y, 1] > threshold and data[t, x, y, 0] < 1 and data[t, x, y, 1] < 1):
                        labels.append(data[t, x, y, :].flatten())
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
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                # get features
                                cnn_feature[cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size),
                                :] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, :]
                        cnn_features[seqn].append(cnn_feature)

    for i in range(short_term_lstm_seq_len):
        cnn_features[i] = np.array(cnn_features[i])

    labels = np.array(labels)

    batch_data_size = 4
    datas_queue = tf.transpose(tf.convert_to_tensor(np.array(cnn_features)),
                               perm=[1, 0, 2, 3, 4])  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
    labels_queue = tf.convert_to_tensor(np.array(labels))  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
    data = tf.data.Dataset.from_tensor_slices(datas_queue)
    label = tf.data.Dataset.from_tensor_slices(labels_queue)

    data = data.batch(batch_data_size)
    label = label.batch(batch_data_size)
    data_iterator = data.make_one_shot_iterator()
    label_iterator = label.make_one_shot_iterator()
    data = data_iterator.get_next()
    label = label_iterator.get_next()
    print('fine-tune or test data:', data)
    print('fine-tune or test label:', label)
    return tf.cast(data, tf.float32), tf.cast(label, tf.float32)

def main():
    mode = args.mode
    short_term_seq_len=7
    kshot = 1
    kquery = 4
    nway = 5
    meta_batchsz = 32
    K = 5
    iterations_pre = 2000
    iterations = 1000



################################
    #SOM_MAML without attention
    db_with_attention = DataGenerator_SOM_MAML_with_attention(nway, kshot, kquery, meta_batchsz)

    data_tensor, label_tensor = db_with_attention.make_data_tensor(mode='pretrain-NYtaxi', total_batch_num=meta_batchsz*iterations_pre)
    support_x_pretrain = tf.slice(data_tensor, [0, 0, 0, 0, 0, 0], [-1,  nway * kshot, -1, -1, -1, -1], name='support_x_pretrain')
    query_x_pretrain = tf.slice(data_tensor, [0,  nway * kshot, 0, 0, 0, 0], [-1, -1, -1, -1, -1, -1], name='query_x_pretrain')
    support_y_pretrain = tf.slice(label_tensor, [0, 0, 0], [-1,  nway * kshot, -1], name='support_y_pretrain')
    query_y_pretrain = tf.slice(label_tensor, [0,  nway * kshot, 0], [-1, -1, -1], name='query_y_pretrain')

    #x_fine_tune_NYbike, y_fine_tune_NYbike = db_with_attention.make_data_tensor(mode='fine-tune_NYbike')
    #x_test_NYbike, y_test_NYbike = db_with_attention.make_data_tensor(mode='NYbike-test')
    x_fine_tune_SZtaxi, y_fine_tune_SZtaxi = db_with_attention.make_data_tensor(mode='fine-tune_SZtaxi')
    x_test_SZtaxi, y_test_SZtaxi = shenzhen_SZ_test()
    #print('-------qvdiao------')
    #print(np.array(x_test_SZtaxi).shape)
    #x_test_SZtaxi, y_test_SZtaxi = db_with_attention.make_data_tensor(mode='SZtaxi-test')
    #print('--------buqv-------')
    #print(np.array(x_test_SZtaxi).shape)





    # 1. construct MAML model
    #modelNYbike_MAML = MAML(short_term_seq_len, 3, 2, nway)
    modelSZtaxi_MAML = MAML(short_term_seq_len, 3, 2, nway)
    #modelNYbike = NO_MAML(short_term_seq_len, 3, 2)
    modelSZtaxi = NO_MAML(short_term_seq_len, 3, 2)



    # construct metatrain_ and metaval
    # NYbike + SOM_MAML
    #modelNYbike_MAML.pretrain(support_x_pretrain, support_y_pretrain, query_x_pretrain, query_y_pretrain, K, meta_batchsz)
    #modelNYbike_MAML.fine_tune(x_fine_tune_NYbike, y_fine_tune_NYbike, x_test_NYbike, y_test_NYbike)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sessNYbikeSOM_MAML = tf.InteractiveSession(config=config)
    # tf.global_variables() to save moving_mean and moving variance of batch norm
    # tf.trainable_variables()  NOT include moving_mean and moving_variance.
    #saverNYbikeSOM_MAML = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # initialize, under interative session
    #tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()

    #if os.path.exists(os.path.join('ckpt/SOM_NYbike', 'checkpoint')):
    #    model_file = tf.train.latest_checkpoint('ckpt/SOM_NYbike')
    #    print("Restoring model weights from ", model_file)
    #    saverNYbikeSOM_MAML.restore(sessNYbikeSOM_MAML, model_file)

    #pretrain(modelNYbike_MAML, saverNYbikeSOM_MAML, sessNYbikeSOM_MAML, iterations_pre)
    #fine_tune(modelNYbike_MAML, saverNYbikeSOM_MAML, sessNYbikeSOM_MAML, iterations)
    #sessNYbikeSOM_MAML.close()



    # SZtaxi + SOM_MAML
    modelSZtaxi_MAML.pretrain(support_x_pretrain, support_y_pretrain, query_x_pretrain, query_y_pretrain, K, meta_batchsz)
    modelSZtaxi_MAML.fine_tune(x_fine_tune_SZtaxi, y_fine_tune_SZtaxi, x_test_SZtaxi, y_test_SZtaxi)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sessSZtaxiSOM_MAML = tf.InteractiveSession(config=config)
    # tf.global_variables() to save moving_mean and moving variance of batch norm
    # tf.trainable_variables()  NOT include moving_mean and moving_variance.
    saverSZtaxiSOM_MAML = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # initialize, under interative session
    tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()

    if os.path.exists(os.path.join('ckpt/SOM_NYbike', 'checkpoint')):
        model_file = tf.train.latest_checkpoint('ckpt/SOM_SZtaxi')
        print("Restoring model weights from ", model_file)
        saverSZtaxiSOM_MAML.restore(sessSZtaxiSOM_MAML, model_file)

    pretrain(modelSZtaxi_MAML, saverSZtaxiSOM_MAML, sessSZtaxiSOM_MAML, iterations_pre)
    fine_tune(modelSZtaxi_MAML, saverSZtaxiSOM_MAML, sessSZtaxiSOM_MAML, iterations)
    sessSZtaxiSOM_MAML.close()

    # NYbike
    #modelNYbike.train(x_fine_tune_NYbike, y_fine_tune_NYbike, x_test_NYbike, y_test_NYbike)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sessNYbike = tf.InteractiveSession(config=config)
    # tf.global_variables() to save moving_mean and moving variance of batch norm
    # tf.trainable_variables()  NOT include moving_mean and moving_variance.
    #saverNYbike = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # initialize, under interative session
    #tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()

    #model_file = tf.train.latest_checkpoint('ckpt/NYbike')
    #print("Restoring model weights from ", model_file)
    #saverNYbike.restore(sessNYbike, model_file)

    #train_without_pretrain(modelNYbike, saverNYbike, sessNYbike, iterations)




    # SZtaxi
    modelSZtaxi.train(x_fine_tune_SZtaxi, y_fine_tune_SZtaxi, x_test_SZtaxi, y_test_SZtaxi)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sessSZtaxi = tf.InteractiveSession(config=config)
    # tf.global_variables() to save moving_mean and moving variance of batch norm
    # tf.trainable_variables()  NOT include moving_mean and moving_variance.
    saverSZtaxi = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # initialize, under interative session
    tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()

    #model_file = tf.train.latest_checkpoint('ckpt/SZtaxi')
    #print("Restoring model weights from ", model_file)
    #saverSZtaxi.restore(sessSZtaxi, model_file)

    train_without_pretrain(modelSZtaxi, saverSZtaxi, sessSZtaxi, iterations)



if __name__ == "__main__":
    main()