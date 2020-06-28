import numpy as np
import os
import random
import tensorflow as tf
import tqdm
import data_loader
#import pickle


def sample_weighted(list, number, weight):
    _total = sum(weight)
    outlist = []
    for i in range(number):
        _random = random.uniform(0, _total)
        for j in range(len(weight)):
            if _random <= sum(weight[:j]):
                outlist.append(list[j])
                break
    return outlist


#return [np.array[14, len(all_data)*nb_samples, 7, 7, 2], np.array[len(all_data)*nb_samples, 2]]
def sampler(data, short_term_lstm_seq_len=7, cnn_nbhd_size=1, threshold=0.05):
    max = np.max(data)
    data = data / 500
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
        if t % 100 == 0:
            print("Now sampling at {0} timeslots.".format(t))
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):

                # label
                if (data[t, x, y, 0] > threshold and data[t, x, y, 1] > threshold and data[
                    t, x, y, 0] < 1 and data[t, x, y, 1] < 1):
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

    return cnn_features, labels

def get_datas(paths, nb_samples=None, training = True):
    if training:
        all_data = [np.load(path, allow_pickle=True)
                    for path in paths]

        sampler = [random.sample(range(data[1].shape[0]), nb_samples)
                    for data in all_data]

        datas = [data[0][:,sampler[i],:,:,:]
                  for data, i in zip(all_data, np.arange(len(all_data)))]
        labels = [data[1][sampler[i],:]
                 for data, i in zip(all_data, np.arange(len(all_data)))]
        data = [np.concatenate(np.array(datas), axis=1), np.concatenate(np.array(labels), axis=0)]
    else:
        all_data = np.load(paths[0], allow_pickle=True)
        sampler = random.sample(range(all_data[1].shape[0]), nb_samples)
        datas = np.array(all_data[0])[:, sampler, :, :, :]
        labels = np.array(all_data[1])[sampler, :]
        data = [datas, labels]
    return data

class DataGenerator_SOM_MAML_without_attention:
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, nway, kshot, kquery, meta_batchsz):
        """

        :param nway:
        :param kshot:
        :param kquery:
        :param meta_batchsz:
        """
        self.meta_batchsz = meta_batchsz #4 task
        # number of images to sample per class
        self.nimg = kshot + kquery # 1 + 15
        self.nway = nway #5
        #self.onestep_datatasz = (7, 3, 3, 2) #onestep_data

        metapretrain_folder = './Data/Pretrain_data_NYtaxi/class_data/'
        metafinetune_folder_NYbike = './Data/Finetune_data_NYbike/'
        metafinetune_folder_SZtaxi = './Data/Finetune_data_SZtaxi/'
        metatest_folder_NYbike = './Data/Finetune_data_NYbike/'
        metatest_folder_SZtaxi = './Data/Finetune_data_SZtaxi/'

        #metatrain_classname_list ['0class',...]
        self.metapretrain_datas = [os.path.join(metapretrain_folder, label) \
                             for label in os.listdir(metapretrain_folder) \
                        ]

        self.metaNYbike_datas = [os.path.join(metafinetune_folder_NYbike, label) \
                           for label in os.listdir(metafinetune_folder_NYbike) \
                        ]

        self.metaSZtaxi_datas = [os.path.join(metafinetune_folder_SZtaxi, label) \
                                 for label in os.listdir(metafinetune_folder_SZtaxi) \
                        ]

        self.metatestNYbike_datas = [os.path.join(metatest_folder_NYbike, label) \
                                 for label in os.listdir(metatest_folder_NYbike) \
                        ]

        self.metatestSZtaxi_datas = [os.path.join(metatest_folder_SZtaxi, label) \
                                     for label in os.listdir(metatest_folder_SZtaxi) \
                                     ]


    def make_data_tensor(self, mode='pretrain-NYtaxi', total_batch_num = 200000):
        """

        :param training:
        :return:
        """
        print('mode:', mode)
        if mode=='pretrain-NYtaxi' or mode=='pretrain-SZtaxi':
            print('make pretrain data tensor ......')
            folders = self.metapretrain_datas# ['0class',...]
            num_total_batches = total_batch_num#2000000
        elif mode=='fine-tune_NYbike':#fine-tune or train_without_pretrain
            print('make fine-tune data tensor ......')
            folders = self.metaNYbike_datas#[0]
            num_total_batches = 600
        elif mode=='fine-tune_SZtaxi':#fine-tune or train_without_pretrain
            print('make fine-tune data tensor ......')
            folders = self.metaSZtaxi_datas#[0]
            num_total_batches = 600
        elif mode=='NYbike-test':
            print('make test data tensor ......')
            folders = self.metatestNYbike_datas#[1]
            num_total_batches = 200
        elif mode=='SZtaxi-test':
            print('make test data tensor ......')
            folders = self.metatestSZtaxi_datas#[1]
            num_total_batches = 200


        if mode=='pretrain-NYtaxi':
            # 16 in one class, 16*5 in one task
            # [task1_0_img0, task1_0_img15, task1_1_img0,]
            all_datas = []
            all_labels = []
            for _ in tqdm.tqdm(range(num_total_batches), 'generating pretrain episodes'):
                # from image folder sample 5 class randomly
                sampled_folders = random.sample(folders, self.nway)#['aclass', 'bclass', 'cclass'...]len=self.nway(5)
                random.shuffle(sampled_folders)
                # sample 16 datas and labels from selected folders
                # len: 5 * 16
                labels_and_datas = get_datas(sampled_folders, nb_samples=self.nimg)

                # make sure the above isn't randomized order
                datas = labels_and_datas[0]
                #print('datas:'+str(np.array(datas).shape))
                labels = labels_and_datas[1]
                #print('labels:' + str(np.array(labels).shape))
                all_datas.extend([datas])#[datas,...]len=2000000 datas:np.array(14,self.nimg*self.nway,7,7,2)
                all_labels.extend([labels])#[labels,...]len=2000000 labels:np.array(self.nimg*self.nway,2)


        elif mode=='fine-tune_NYbike':
            path = folders[0]
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))

        elif mode=='fine-tune_SZtaxi' :
            path = folders[6]
            print(folders)
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))

        elif mode == 'NYbike-test':# test
            path = folders[4]
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))
        elif mode == 'SZtaxi-test':# test
            path = folders[1]
            print(path)
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))

        if mode=='pretrain-NYtaxi':
            # make queue for tensorflow to read from
            print('creating pipeline ops')
            print(all_datas[0].shape)
            print(np.array(all_labels).shape)
            datas_queue = tf.transpose(tf.convert_to_tensor(np.concatenate(all_datas, axis=1)), perm=[1,0,2,3,4])#tensor(2000000*self.nimg*self.nway,14,7,7,2)
            labels_queue = tf.convert_to_tensor(np.concatenate(all_labels, axis=0))#tensor(2000000*self.nimg*self.nway,14,7,7,2)


            # tensorflow format: N*short_time_seq_len*H*W*C


            examples_per_batch = self.nway * self.nimg   # 2*4
            # batch here means batch of meta-learning, including 4 tasks = 4*8
            batch_data_size = self.meta_batchsz * examples_per_batch # 4*8

            data=tf.data.Dataset.from_tensor_slices(datas_queue)
            label=tf.data.Dataset.from_tensor_slices(labels_queue)
            print('batching images')
            #data, label = tf.train.batch(
            #	[datas_queue, labels_queue],
            #	batch_size=batch_data_size, # 4*80
            #	num_threads=self.meta_batchsz,
            #	capacity=256+ 14 * batch_data_size, # 256 + 3* 4*80
            #)
            #print(data)
            #print(label)

            data = data.batch(batch_data_size)
            label=label.batch(batch_data_size)
            data_iterator = data.make_one_shot_iterator()
            label_iterator = label.make_one_shot_iterator()
            data = data_iterator.get_next()
            label = label_iterator.get_next()



            all_data_batches, all_label_batches = [], []
            print('manipulating images to be right order')

            # images contains current batch, namely 4 task, 4* 80
            for i in range(self.meta_batchsz): # 4
                # current task, 80 images
                data_batch = data[i * examples_per_batch:(i + 1) * examples_per_batch]#(80,14,7,7,2)
                label_batch = label[i * examples_per_batch:(i + 1) * examples_per_batch]#(80,14,7,7,2)

                new_list, new_label_list = [], []
                # for each image from 0 to 15 in all 5 class
                for k in range(self.nimg): # 16
                    class_idxs = tf.range(0, self.nway) # [0,1,2,3,4]
                    class_idxs = tf.random_shuffle(class_idxs)#[3,2,4,0,1]
                    # it will cope with 5 images parallelly
                    #    [0, 16, 32, 48, 64] or [1, 17, 33, 49, 65]
                    true_idxs = class_idxs * self.nimg + k
                    new_list.append(tf.gather(data_batch, true_idxs))
                    new_label_list.append(tf.gather(label_batch, true_idxs))

                # [80, 14*7*7*2]
                new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
                # [80,2]
                new_label_list = tf.concat(new_label_list, 0)
                all_data_batches.append(new_list)
                all_label_batches.append(new_label_list)

            # [4, 80, 14*7*7*2]
            all_data_batches = tf.cast(tf.stack(all_data_batches), tf.float32)
            # [4, 80, 2]
            all_label_batches = tf.cast(tf.stack(all_label_batches), tf.float32)


            #print('data_b:', all_data_batches)
            #print('label_b:', all_label_batches)

            return all_data_batches, all_label_batches

        elif mode == 'NYbike-test' or mode == 'SZtaxi-test':
            all_datas = tf.transpose(tf.convert_to_tensor(np.array(all_datas)), perm=[1, 0, 2, 3, 4])  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
            all_labels = tf.convert_to_tensor(np.array(all_labels))
            return tf.cast(all_datas, tf.float32), tf.cast(all_labels, tf.float32)

        else:
            batch_data_size = 256
            datas_queue = tf.transpose(tf.convert_to_tensor(np.array(all_datas)), perm=[1, 0, 2, 3, 4])  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
            labels_queue = tf.convert_to_tensor(np.array(all_labels))  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
            data = tf.data.Dataset.from_tensor_slices(datas_queue)
            label = tf.data.Dataset.from_tensor_slices(labels_queue)

            data = data.batch(batch_data_size)
            data = data.repeat()
            label = label.batch(batch_data_size)
            label = label.repeat()
            data_iterator = data.make_one_shot_iterator()
            label_iterator = label.make_one_shot_iterator()
            data = data_iterator.get_next()
            label = label_iterator.get_next()
            #print('fine-tune or test data:', data)
            #print('fine-tune or test label:', label)
            return tf.cast(data, tf.float32), tf.cast(label, tf.float32)


class DataGenerator_SOM_MAML_with_attention:
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, nway, kshot, kquery, meta_batchsz):
        """

        :param nway:
        :param kshot:
        :param kquery:
        :param meta_batchsz:
        """
        self.meta_batchsz = meta_batchsz #4 task
        # number of images to sample per class
        self.nimg = kshot + kquery # 1 + 15
        self.nway = nway #5
        #self.onestep_datatasz = (7, 3, 3, 2) #onestep_data

        metapretrain_folder = './Data/Pretrain_data_NYtaxi/class_data/'
        metafinetune_folder_NYbike = './Data/Finetune_data_NYbike/'
        metafinetune_folder_SZtaxi = './Data/Finetune_data_SZtaxi/'
        metatest_folder_NYbike = './Data/Finetune_data_NYbike/'
        metatest_folder_SZtaxi = './Data/Finetune_data_SZtaxi/'

        #metatrain_classname_list ['0class',...]
        self.metapretrain_datas = [os.path.join(metapretrain_folder, label) \
                             for label in os.listdir(metapretrain_folder) \
                        ]

        self.metaNYbike_datas = [os.path.join(metafinetune_folder_NYbike, label) \
                           for label in os.listdir(metafinetune_folder_NYbike) \
                        ]

        self.metaSZtaxi_datas = [os.path.join(metafinetune_folder_SZtaxi, label) \
                                 for label in os.listdir(metafinetune_folder_SZtaxi) \
                        ]

        self.metatestNYbike_datas = [os.path.join(metatest_folder_NYbike, label) \
                                 for label in os.listdir(metatest_folder_NYbike) \
                        ]

        self.metatestSZtaxi_datas = [os.path.join(metatest_folder_SZtaxi, label) \
                                     for label in os.listdir(metatest_folder_SZtaxi) \
                                     ]


    def make_data_tensor(self, mode='pretrain-NYtaxi', total_batch_num = 200000):
        """

        :param training:
        :return:
        """
        print('mode:', mode)
        if mode=='pretrain-NYtaxi' or mode=='pretrain-SZtaxi':
            print('make pretrain data tensor ......')
            folders = self.metapretrain_datas# ['0class',...]
            num_total_batches = total_batch_num#2000000
        elif mode=='fine-tune_NYbike':#fine-tune or train_without_pretrain
            print('make fine-tune data tensor ......')
            folders = self.metaNYbike_datas#[0]
            num_total_batches = 600
        elif mode=='fine-tune_SZtaxi':#fine-tune or train_without_pretrain
            print('make fine-tune data tensor ......')
            folders = self.metaSZtaxi_datas#[0]
            num_total_batches = 600
        elif mode=='NYbike-test':
            print('make test data tensor ......')
            folders = self.metatestNYbike_datas#[1]
            num_total_batches = 200
        elif mode=='SZtaxi-test':
            print('make test data tensor ......')
            folders = self.metatestSZtaxi_datas#[1]
            num_total_batches = 200


        if mode=='pretrain-NYtaxi':
            # 16 in one class, 16*5 in one task
            # [task1_0_img0, task1_0_img15, task1_1_img0,]
            all_datas = []
            all_labels = []
            print(folders)
            SZtaxiweight = []
            for folder in folders:
                SZtaxiweight.append(np.load(folder, allow_pickle=True)[1].shape[0])

            print(SZtaxiweight)

            for _ in tqdm.tqdm(range(num_total_batches), 'generating pretrain episodes'):
                # from image folder sample 5 class randomly
                sampled_folders = sample_weighted(folders, self.nway, SZtaxiweight)#['aclass', 'bclass', 'cclass'...]len=self.nway(5)
                random.shuffle(sampled_folders)
                # sample 16 datas and labels from selected folders
                # len: 5 * 16
                labels_and_datas = get_datas(sampled_folders, nb_samples=self.nimg)

                # make sure the above isn't randomized order
                datas = labels_and_datas[0]
                #print('datas:'+str(np.array(datas).shape))
                labels = labels_and_datas[1]
                #print('labels:' + str(np.array(labels).shape))
                all_datas.extend([datas])#[datas,...]len=2000000 datas:np.array(14,self.nimg*self.nway,7,7,2)
                all_labels.extend([labels])#[labels,...]len=2000000 labels:np.array(self.nimg*self.nway,2)


        elif mode=='fine-tune_NYbike':
            path = folders[0]
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))

        elif mode=='fine-tune_SZtaxi' :
            path = folders[6]
            print(folders)
            print('------path--------')
            print(path)
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))

        elif mode == 'NYbike-test':# test
            path = folders[4]
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))
        elif mode == 'SZtaxi-test':# test
            path = folders[1]
            print(path)
            all_datas, all_labels = sampler(np.load(path, allow_pickle=True))

        if mode=='pretrain-NYtaxi':
            # make queue for tensorflow to read from
            print('creating pipeline ops')
            print(all_datas[0].shape)
            print(np.array(all_labels).shape)
            datas_queue = tf.transpose(tf.convert_to_tensor(np.concatenate(all_datas, axis=1)), perm=[1,0,2,3,4])#tensor(2000000*self.nimg*self.nway,14,7,7,2)
            labels_queue = tf.convert_to_tensor(np.concatenate(all_labels, axis=0))#tensor(2000000*self.nimg*self.nway,14,7,7,2)


            # tensorflow format: N*short_time_seq_len*H*W*C


            examples_per_batch = self.nway * self.nimg   # 2*4
            # batch here means batch of meta-learning, including 4 tasks = 4*8
            batch_data_size = self.meta_batchsz * examples_per_batch # 4*8

            data=tf.data.Dataset.from_tensor_slices(datas_queue)
            label=tf.data.Dataset.from_tensor_slices(labels_queue)
            print('batching images')
            #data, label = tf.train.batch(
            #	[datas_queue, labels_queue],
            #	batch_size=batch_data_size, # 4*80
            #	num_threads=self.meta_batchsz,
            #	capacity=256+ 14 * batch_data_size, # 256 + 3* 4*80
            #)
            #print(data)
            #print(label)

            data = data.batch(batch_data_size)
            label=label.batch(batch_data_size)
            data_iterator = data.make_one_shot_iterator()
            label_iterator = label.make_one_shot_iterator()
            data = data_iterator.get_next()
            label = label_iterator.get_next()



            all_data_batches, all_label_batches = [], []
            print('manipulating images to be right order')

            # images contains current batch, namely 4 task, 4* 80
            for i in range(self.meta_batchsz): # 4
                # current task, 80 images
                data_batch = data[i * examples_per_batch:(i + 1) * examples_per_batch]#(80,14,7,7,2)
                label_batch = label[i * examples_per_batch:(i + 1) * examples_per_batch]#(80,14,7,7,2)

                new_list, new_label_list = [], []
                # for each image from 0 to 15 in all 5 class
                for k in range(self.nimg): # 16
                    class_idxs = tf.range(0, self.nway) # [0,1,2,3,4]
                    class_idxs = tf.random_shuffle(class_idxs)#[3,2,4,0,1]
                    # it will cope with 5 images parallelly
                    #    [0, 16, 32, 48, 64] or [1, 17, 33, 49, 65]
                    true_idxs = class_idxs * self.nimg + k
                    new_list.append(tf.gather(data_batch, true_idxs))
                    new_label_list.append(tf.gather(label_batch, true_idxs))

                # [80, 14*7*7*2]
                new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
                # [80,2]
                new_label_list = tf.concat(new_label_list, 0)
                all_data_batches.append(new_list)
                all_label_batches.append(new_label_list)

            # [4, 80, 14*7*7*2]
            all_data_batches = tf.cast(tf.stack(all_data_batches), tf.float32)
            # [4, 80, 2]
            all_label_batches = tf.cast(tf.stack(all_label_batches), tf.float32)


            #print('data_b:', all_data_batches)
            #print('label_b:', all_label_batches)

            return all_data_batches, all_label_batches

        elif mode == 'NYbike-test' or mode == 'SZtaxi-test':
            all_datas = tf.transpose(tf.convert_to_tensor(np.array(all_datas)), perm=[1, 0, 2, 3, 4])  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
            all_labels = tf.convert_to_tensor(np.array(all_labels))
            return tf.cast(all_datas, tf.float32), tf.cast(all_labels, tf.float32)

        else:
            batch_data_size = 4
            datas_queue = tf.transpose(tf.convert_to_tensor(np.array(all_datas)), perm=[1, 0, 2, 3, 4])  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
            labels_queue = tf.convert_to_tensor(np.array(all_labels))  # tensor(2000000*self.nimg*self.nway,14,7,7,2)
            data = tf.data.Dataset.from_tensor_slices(datas_queue)
            label = tf.data.Dataset.from_tensor_slices(labels_queue)

            data = data.batch(batch_data_size)
            data = data.repeat()
            label = label.batch(batch_data_size)
            label = label.repeat()
            data_iterator = data.make_one_shot_iterator()
            label_iterator = label.make_one_shot_iterator()
            data = data_iterator.get_next()
            label = label_iterator.get_next()
            #print('fine-tune or test data:', data)
            #print('fine-tune or test label:', label)
            return tf.cast(data, tf.float32), tf.cast(label, tf.float32)
