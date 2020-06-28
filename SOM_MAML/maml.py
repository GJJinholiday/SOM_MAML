import numpy as np
import tensorflow as tf
#from keras.models import Model
#from keras.layers import Dense, Activation, Input, LSTM, Conv2D, Flatten, Concatenate, Reshap
from tensorflow._api.v1.nn.rnn_cell import RNNCell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs


class NO_MAML:
    def __init__(self, l, d, c, train_lr=1e-3):
        """

        :param d:
        :param c:
        :param nway:
        :param meta_lr:
        :param train_lr:
        """
        self.l = l
        self.d = d
        self.c = c
        self.train_lr = train_lr

        print('img shape:', self.l, self.d, self.d, self.c, 'train-lr:', train_lr)

    def train(self, x_fine_tune, y_fine_tune, x_test, y_test):
        self.weights = self.get_weights()
        x_fine_tune_pred = self.forward(x_fine_tune, self.weights)
        self.finetune_loss = tf.losses.mean_squared_error(x_fine_tune_pred, y_fine_tune)
        optimizer = tf.train.AdamOptimizer(self.train_lr, name='train_optim')
        # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
        gvs = optimizer.compute_gradients(self.finetune_loss)
        # meta-train grads clipping
        gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
        # update theta
        self.train_op = optimizer.apply_gradients(gvs)
        # use theta_pi to forward meta-test
        x_test_pred = self.forward(x_test, self.weights)
        # meta-test loss
        self.loss = tf.reduce_sum(tf.losses.mean_squared_error(x_test_pred, y_test))
        self.mape = tf.reduce_mean(tf.div(tf.losses.absolute_difference(x_test_pred, y_test), y_test))



    def get_weights(self):
        weights = {}

        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        k = 3

        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            weights['conv1'] = tf.get_variable('conv1w', [self.l, k, k, 2, 32], initializer=conv_initializer)
            weights['b1'] = tf.get_variable('conv1b', initializer=tf.zeros([self.l, 32]))
            weights['conv2'] = tf.get_variable('conv2w', [self.l, k, k, 32, 32], initializer=conv_initializer)
            weights['b2'] = tf.get_variable('conv2b', initializer=tf.zeros([self.l, 32]))
            weights['conv3'] = tf.get_variable('conv3w', [self.l, k, k, 32, 32], initializer=conv_initializer)
            weights['b3'] = tf.get_variable('conv3b', initializer=tf.zeros([self.l, 32]))
            weights['conv4'] = tf.get_variable('conv4w', [self.l, k, k, 32, 32], initializer=conv_initializer)
            weights['b4'] = tf.get_variable('conv4b', initializer=tf.zeros([self.l, 32]))
            weights['dense'] = tf.get_variable('dense', [self.l, 3 * 3 * 32, 128], initializer=conv_initializer)
            weights['fc'] = tf.get_variable('fc', [128, 2], initializer=conv_initializer)
            weights['hidden0'] = tf.get_variable('hidden0', [7 * 128, 128], initializer=conv_initializer)
            for i in range(7):
                weights['RNN' + str(i)] = tf.get_variable('RNN' + str(i), [128, 128], initializer=conv_initializer)

            return weights



    def conv_block(self, x, weight, bias, scope):
        """
        build a block with conv2d->batch_norm->pooling
        :param x:
        :param weight:
        :param bias:
        :param scope:
        :param training:
        :return:
        """
        # conv
        x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias
        # batch norm, activation_fn=tf.nn.relu,
        # NOTICE: must have tf.layers.batch_normalization
        # x = tf.contrib.layers.batch_norm(x, activation_fn=tf.nn.relu)
        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            # train is set to True ALWAYS, please refer to https://github.com/cbfinn/maml/issues/9
            # when FLAGS.train=True, we still need to build evaluation network
            x = tf.layers.batch_normalization(x, training=True, name=scope + '_bn', reuse=tf.AUTO_REUSE)
        # relu
        x = tf.nn.relu(x, name=scope + '_relu')
        # pooling
        #x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=scope + '_pool')
        return x








# ------------------AugmentedConv-------------------#
    def forward(self, x, weights):
        """


        :param x:
        :param weights:
        :param training:
        :return:
        """
        hidden2 = [self.conv_block(tf.squeeze(x[:, i, :, :, :]), weights['conv1'][i], weights['b1'][i], 'conv0') for i
                   in
                   range(self.l)]
        hidden3 = [self.conv_block(tf.squeeze(hidden2[i]), weights['conv2'][i], weights['b2'][i], 'conv1') for i in
                   range(self.l)]
        hidden4 = [self.conv_block(tf.squeeze(hidden3[i]), weights['conv3'][i], weights['b3'][i], 'conv2') for i in
                   range(self.l)]
        hidden5 = [self.conv_block(tf.squeeze(hidden4[i]), weights['conv4'][i], weights['b4'][i], 'conv3') for i in
                   range(self.l)]
        hidden6 = [tf.layers.flatten(hidden5[i]) for i in range(self.l)]
        hidden7 = [tf.matmul(hidden6[i], weights['dense'][i]) for i in range(self.l)]
        hidden = tf.matmul(tf.concat(hidden7, axis=1), weights['hidden0'])
        hidden = tf.sigmoid(hidden)
        for i in range(len(hidden7)):
            hidden = tf.sigmoid(tf.matmul(hidden7[i], weights['RNN' + str(i)])) + tf.sigmoid(hidden)
        output = hidden
        output = tf.matmul(output, weights['fc'])
        output = tf.sigmoid(output)

        return output

class MAML:
    def __init__(self, l, d, c, nway, meta_lr=1e-2, train_lr=1e-3):
        """

        :param d:
        :param c:
        :param nway:
        :param meta_lr:
        :param train_lr:
        """
        self.l = l
        self.d = d
        self.c = c
        self.nway = nway
        self.meta_lr = meta_lr
        self.train_lr = train_lr

        print('img shape:', self.l, self.d, self.d, self.c, 'meta-lr:', meta_lr, 'train-lr:', train_lr)

    def pretrain(self, support_xb, support_yb, query_xb, query_yb, K, meta_batchsz):
        """

        :param support_xb:   [b, setsz, 14*7*7*2]
        :param support_yb:   [b, setsz, 2]
        :param query_xb:     [b, querysz, 14*7*7*2]
        :param query_yb:     [b, querysz, 2]
        :param K:            train update steps
        :param meta_batchsz: tasks number
        :param mode:         train/eval/test, for training, we build train&eval network meanwhile.
        :return:
        """
        # create or reuse network variable, not including batch_norm variable, therefore we need extra reuse mechnism
        # to reuse batch_norm variables.
        self.weights = self.get_weights()
        # TODO: meta-test is sort of test stage.

        def meta_task(input):
            """
            map_fn only support one parameters, so we need to unpack from tuple.
            :param support_x:   [setsz, 14*7*7*2]
            :param support_y:   [setsz, 2]
            :param query_x:     [querysz, 14*7*7*2]
            :param query_y:     [querysz, 2]
            :param training:    training or not, for batch_norm
            :return:
            """
            support_x, support_y, query_x, query_y = input
            # to record the op in t update step.
            query_preds, query_losses, query_abs_losses = [], [], []

            # ==================================
            # REUSE       True        False
            # Not exist   Error       Create one
            # Existed     reuse       Error
            # ==================================
            # That's, to create variable, you must turn off reuse
            support_pred = self.forward(support_x, self.weights)
            support_loss = tf.losses.mean_squared_error(support_pred, support_y)
            #print('support_loss', support_loss)
            # compute gradients
            #print(list(self.weights.values()))
            grads = tf.gradients(support_loss, list(self.weights.values()))
            # grad and variable dict
            gvs = dict(zip(self.weights.keys(), grads))
            print(gvs)

            # theta_pi = theta - alpha * grads
            fast_weights = dict(zip(self.weights.keys(),
                                    [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
            # use theta_pi to forward meta-test
            query_pred = self.forward(query_x, fast_weights)
            # meta-test loss
            query_loss = tf.losses.mean_squared_error(query_pred, query_y)
            query_abs_loss = tf.losses.absolute_difference(query_pred, query_y)
            # record T0 pred and loss for meta-test
            query_preds.append(query_pred)
            query_losses.append(query_loss)
            query_abs_losses.append(query_abs_loss)

            # continue to build T1-TK-1 steps graph
            for _ in range(1, K):
                # T_k loss on meta-train
                # we need meta-train loss to fine-tune the task and meta-test loss to update theta
                loss = tf.losses.mean_squared_error(self.forward(support_x, fast_weights), support_y)
                # compute gradients
                grads = tf.gradients(loss, list(fast_weights.values()))
                # compose grad and variable dict
                gvs = dict(zip(fast_weights.keys(), grads))
                # update theta_pi according to varibles
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key]
                                                              for key in fast_weights.keys()]))
                # forward on theta_pi
                query_pred = self.forward(query_x, fast_weights)
                # we need accumulate all meta-test losses to update theta
                query_loss = tf.losses.mean_squared_error(query_pred, query_y)
                query_abs_loss = tf.losses.absolute_difference(query_pred, query_y)
                query_preds.append(query_pred)
                query_losses.append(query_loss)
                query_abs_losses.append(query_abs_loss)


            # we just use the first step support op: support_pred & support_loss, but igonre these support op
            # at step 1:K-1.
            # however, we return all pred&loss&acc op at each time steps.
            result = [support_pred, support_loss, query_preds, query_losses, query_abs_losses]

            return result
            # return: [support_pred, support_loss, query_preds, query_losses]

        out_dtype = [tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
        result = tf.map_fn(meta_task, elems=(support_xb, support_yb, query_xb, query_yb),
                           dtype=out_dtype, parallel_iterations=meta_batchsz, name='map_fn')
        support_pred_tasks, support_loss_tasks, \
        query_preds_tasks, query_losses_tasks, query_abs_losses_tasks = result
        #self.y =  query_yb
        #self.p = query_losses_tasks

        # average loss
        self.support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
        # [avgloss_t1, avgloss_t2, ..., avgloss_K]
        self.query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                for j in range(K)]

        self.query_mapes = [tf.reduce_mean(query_abs_losses_tasks[j]) / tf.reduce_mean(query_yb)
                                                for j in range(K)]

        # # add batch_norm ops before meta_op
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        # TODO: the update_ops must be put before tf.train.AdamOptimizer,
        # otherwise it throws Not in same Frame Error.
        # meta_loss = tf.identity(self.query_losses[-1])

        # meta-train optim
        optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
        # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
        gvs = optimizer.compute_gradients(self.query_losses[-1])
        # meta-train grads clipping
        gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
        # update theta
        self.meta_op = optimizer.apply_gradients(gvs)


    def fine_tune(self, x_fine_tune, y_fine_tune, x_test, y_test):
        self.weights = self.get_weights()
        x_fine_tune_pred = self.forward(x_fine_tune, self.weights)
        self.finetune_loss = tf.losses.mean_squared_error(x_fine_tune_pred, y_fine_tune)
        optimizer = tf.train.AdamOptimizer(self.train_lr, name='finetune_optim')
        # meta-train gradients, query_losses[-1] is the accumulated loss across over tasks.
        gvs = optimizer.compute_gradients(self.finetune_loss)
        # meta-train grads clipping
        gvs = [(tf.clip_by_norm(grad, 10), var) for grad, var in gvs]
        # update theta
        self.finetune_op = optimizer.apply_gradients(gvs)
        # use theta_pi to forward meta-test
        x_test_pred = self.forward(x_test, self.weights)
        # meta-test loss
        self.loss = tf.reduce_sum(tf.losses.mean_squared_error(x_test_pred, y_test))
        self.mape = tf.reduce_mean(tf.div(tf.losses.absolute_difference(x_test_pred, y_test), y_test))

    def get_weights(self):
        weights = {}

        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        k = 3

        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            weights['conv1'] = tf.get_variable('conv1w', [self.l, k, k, 2, 32], initializer=conv_initializer)
            weights['b1'] = tf.get_variable('conv1b', initializer=tf.zeros([self.l, 32]))
            weights['conv2'] = tf.get_variable('conv2w', [self.l, k, k, 32, 32], initializer=conv_initializer)
            weights['b2'] = tf.get_variable('conv2b', initializer=tf.zeros([self.l, 32]))
            weights['conv3'] = tf.get_variable('conv3w', [self.l, k, k, 32, 32], initializer=conv_initializer)
            weights['b3'] = tf.get_variable('conv3b', initializer=tf.zeros([self.l, 32]))
            weights['conv4'] = tf.get_variable('conv4w', [self.l, k, k, 32, 32], initializer=conv_initializer)
            weights['b4'] = tf.get_variable('conv4b', initializer=tf.zeros([self.l, 32]))
            weights['dense'] = tf.get_variable('dense', [self.l, 3 * 3 * 32, 128], initializer=conv_initializer)
            weights['fc'] = tf.get_variable('fc', [128, 2], initializer=conv_initializer)
            weights['hidden0'] = tf.get_variable('hidden0', [7*128, 128], initializer=conv_initializer)
            for i in range(7):
                weights['RNN' + str(i)] = tf.get_variable('RNN'+str(i), [128, 128], initializer=conv_initializer)
            return weights

    def conv_block(self, x, weight, bias, scope):
        """
        build a block with conv2d->batch_norm->pooling
        :param x:
        :param weight:
        :param bias:
        :param scope:
        :param training:
        :return:
        """
        # conv
        x = tf.nn.conv2d(x, weight, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias
        # batch norm, activation_fn=tf.nn.relu,
        # NOTICE: must have tf.layers.batch_normalization
        # x = tf.contrib.layers.batch_norm(x, activation_fn=tf.nn.relu)
        with tf.variable_scope('MAML', reuse=tf.AUTO_REUSE):
            # train is set to True ALWAYS, please refer to https://github.com/cbfinn/maml/issues/9
            # when FLAGS.train=True, we still need to build evaluation network
            x = tf.layers.batch_normalization(x, training=True, name=scope + '_bn', reuse=tf.AUTO_REUSE)
        # relu
        x = tf.nn.relu(x, name=scope + '_relu')
        # pooling
        #x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID', name=scope + '_pool')
        return x

    def augmentedConv(x, weight1, weight2, weight3, bias1, bias2, bias3, scope, dk, dv, Nh, relative=True):
            def shape_list(x):
                """Return list of dims, statically where possible."""
                static = x.get_shape().as_list()
                shape = tf.shape(x)
                ret = []
                for i, static_dim in enumerate(static):
                    dim = static_dim or shape[i]
                    ret.append(dim)
                return ret

            def split_heads_2d(inputs, Nh):
                """Split channels into multiple heads."""
                B, H, W, d = shape_list(inputs)
                ret_shape = [B, H, W, Nh, d // Nh]
                split = tf.reshape(inputs, ret_shape)
                return tf.transpose(split, [0, 3, 1, 2, 4])

            def combine_heads_2d(inputs):
                """Combine heads (inverse of split heads 2d)."""
                transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
                Nh, channels = shape_list(transposed)[-2:]
                ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
                return tf.reshape(transposed, ret_shape)

            def rel_to_abs(x):
                """Converts tensor from relative to aboslute indexing."""
                # [B, Nh, L, 2L 1]
                B, Nh, L, _ = shape_list(x)
                # Pad to shift from relative to absolute indexing.
                col_pad = tf.zeros((B, Nh, L, 1))
                x = tf.concat([x, col_pad], axis=3)
                flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
                flat_pad = tf.zeros((B, Nh, L - 1))
                flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
                # Reshape and slice out the padded elements.
                final_x = tf.reshape(flat_x_padded, [B, Nh, L + 1, 2 * L - 1])
                final_x = final_x[:, :, :L, L - 1:]
                return final_x

            def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
                """Compute relative logits along one dimenion."""
                rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
                # Collapse height and heads
                rel_logits = tf.reshape(
                    rel_logits, [-1, Nh * H, W, 2 * W - 1])
                rel_logits = rel_to_abs(rel_logits)
                # Shape it and tile height times
                rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
                rel_logits = tf.expand_dims(rel_logits, axis=3)
                rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
                # Reshape for adding to the logits.
                rel_logits = tf.transpose(rel_logits, transpose_mask)
                rel_logits = tf.reshape(rel_logits, [-1, Nh, H * W, H * W])
                return rel_logits

            def relative_logits(q, H, W, Nh, dkh):
                """Compute relative logits."""
                # Relative logits in width dimension first.
                var = np.ones((2 * W - 1, dkh), dtype=np.float32)
                rel_embeddings_w = tf.Variable(var, name='r_width', dtype='float32')
                # [B, Nh, HW, HW]
                rel_logits_w = relative_logits_1d(q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5])
                # Relative logits in height dimension next.
                # For ease, we 1) transpose height and width,
                # 2) repeat the above steps and
                # 3) transpose to eventually put the logits
                # in their right positions.
                var = np.ones((2 * H - 1, dkh), dtype=np.float32)
                rel_embeddings_h = tf.Variable(var, name='r_heigh', dtype='float32')
                # [B, Nh, HW, HW]
                rel_logits_h = relative_logits_1d(
                    tf.transpose(q, [0, 1, 3, 2, 4]),
                    rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
                return rel_logits_h, rel_logits_w

            def self_attention_2d(inputs, weight1, weight2, bias1, bias2, dk, dv, Nh, relative=True):
                """2d relative self attention."""
                _, H, W, _ = shape_list(inputs)
                dkh = dk // Nh
                dvh = dv // Nh
                flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H * W, d])
                # Compute q, k, v
                kqv = tf.nn.conv2d(inputs, weight1, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias1
                k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
                q *= dkh ** -0.5  # scaled dot product
                # After splitting, shape is [B, Nh, H, W, dkh or dvh]
                q = split_heads_2d(q, Nh)
                k = split_heads_2d(k, Nh)
                v = split_heads_2d(v, Nh)
                # [B, Nh, HW, HW]

                logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh), transpose_b=True)
                if relative:
                    rel_logits_h, rel_logits_w = relative_logits(q, H, W, Nh, dkh)
                    logits += rel_logits_h
                    logits += rel_logits_w
                weights = tf.nn.softmax(logits)
                attn_out = tf.matmul(weights, flatten_hw(v, dvh))
                attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
                attn_out = combine_heads_2d(attn_out)
                # Project heads
                attn_out = tf.nn.conv2d(attn_out, weight2, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias2
                return attn_out

            conv_out = tf.nn.conv2d(x, weight3, [1, 1, 1, 1], 'SAME', name=scope + '_conv2d') + bias3

            attn_out = self_attention_2d(x, weight1, weight2, bias1, bias2, dk, dv, Nh, relative=
            relative)

            return tf.concat([conv_out, attn_out], axis=3)

        # ------------------AugmentedConv-------------------#
    def forward(self, x, weights):
            """


            :param x:
            :param weights:
            :param training:
            :return:
            """
            #hidden1 = [self.augmentedConv(tf.squeeze(x[:, i, :, :, :]), weights['aguconv1'], weights['aguconv2'],
            #                              weights['aguconv3'], weights['baguconv1'], weights['baguconv2'],
            #                              weights['baguconv3'], dk=4, dv=4, Nh=4) for i in range(self.l)]
            hidden2 = [self.conv_block(tf.squeeze(x[:, i, :, :, :]), weights['conv1'][i], weights['b1'][i], 'conv0') for i in
                       range(self.l)]
            hidden3 = [self.conv_block(tf.squeeze(hidden2[i]), weights['conv2'][i], weights['b2'][i], 'conv1') for i in
                       range(self.l)]
            hidden4 = [self.conv_block(tf.squeeze(hidden3[i]), weights['conv3'][i], weights['b3'][i], 'conv2') for i in
                       range(self.l)]
            hidden5 = [self.conv_block(tf.squeeze(hidden4[i]), weights['conv4'][i], weights['b4'][i], 'conv3') for i in
                       range(self.l)]
            hidden6 = [tf.layers.flatten(hidden5[i]) for i in range(self.l)]
            hidden7 = [tf.matmul(hidden6[i], weights['dense'][i]) for i in range(self.l)]
            hidden = tf.matmul(tf.concat(hidden7, axis=1), weights['hidden0'])
            hidden = tf.sigmoid(hidden)
            for i in range(len(hidden7)):
                hidden = tf.sigmoid(tf.matmul(hidden7[i], weights['RNN' + str(i)])) + tf.sigmoid(hidden)
            output = hidden
            output = tf.matmul(output, weights['fc'])
            output = tf.sigmoid(output)

            # get_shape is static shape, (5, 5, 5, 32)
            # print('flatten:', hidden4.get_shape())
            # flatten layer

            return output