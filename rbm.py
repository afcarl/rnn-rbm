from __future__ import print_function

import itertools

import pylab
import theano
import theano.tensor as T
import numpy as np
import os
import time
import random

from utils import sigm, dtype, rng, shared_normal, shared_zeros

hidden_scalar = 1.7
hidden_recurrent_scalar = 2.2

theano.config.exception_verbosity = 'high'


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent, lr, l2_norm=0.0001, l1_norm=0.0001):
    # rbm params
    W = shared_normal('W', n_visible, n_hidden, scale=0.01)
    bv = shared_zeros('bv', n_visible)
    bh = shared_zeros('bh', n_hidden)

    # rnn -> rbm connections
    Wuh = shared_normal('Wuh', n_hidden_recurrent, n_hidden, scale=0.0001)
    Wuv = shared_normal('Wuv', n_hidden_recurrent, n_visible, scale=0.0001)

    params = [W, bv, bh, Wuh, Wuv]

    def get_rnn_params(number, n_visible, n_hidden_recurrent):
        w_in_update = shared_normal('w_in_update_%d' % number, n_visible, n_hidden_recurrent, scale=0.0001)
        w_hidden_update = shared_normal('w_hidden_update_%d' % number, n_hidden_recurrent, n_hidden_recurrent,
                                        scale=0.0001)
        b_update = shared_zeros('b_update_%d' % number, n_hidden_recurrent)
        w_in_reset = shared_normal('w_in_reset_%d' % number, n_visible, n_hidden_recurrent, scale=0.0001)
        w_hidden_reset = shared_normal('w_hidden_reset_%d' % number, n_hidden_recurrent, n_hidden_recurrent,
                                       scale=0.0001)
        b_reset = shared_zeros('b_reset_%d' % number, n_hidden_recurrent)
        w_in_hidden = shared_normal('w_in_hidden_%d' % number, n_visible, n_hidden_recurrent, scale=0.0001)
        w_reset_hidden = shared_normal('w_reset_hidden_%d' % number, n_hidden_recurrent, n_hidden_recurrent,
                                       scale=0.0001)
        b_hidden = shared_zeros('b_hidden_%d' % number, n_hidden_recurrent)
        return [w_in_update, w_hidden_update, b_update,
                w_in_reset, w_hidden_reset, b_reset,
                w_in_hidden, w_reset_hidden, b_hidden]

    def build_rnn(params, v_t, u_tm1):
        w_in_update, w_hidden_update, b_update, \
        w_in_reset, w_hidden_reset, b_reset, \
        w_in_hidden, w_reset_hidden, b_hidden = params

        update_gate = T.tanh(T.dot(v_t, w_in_update) + T.dot(u_tm1, w_hidden_update) + b_update)
        reset_gate = T.tanh(T.dot(v_t, w_in_reset) + T.dot(u_tm1, w_hidden_reset) + b_reset)
        u_t_temp = T.tanh(T.dot(v_t, w_in_hidden) + T.dot(u_tm1 * reset_gate, w_reset_hidden) + b_hidden)
        u_t = (1 - update_gate) * u_t_temp + update_gate * u_tm1

        return u_t

    # update gate
    rnn_params_1 = get_rnn_params(1, n_visible, n_hidden_recurrent)
    rnn_params_2 = get_rnn_params(2, n_hidden_recurrent, n_hidden_recurrent)
    rnn_params_3 = get_rnn_params(3, n_hidden_recurrent, n_hidden_recurrent)
    params += rnn_params_1 + rnn_params_2 + rnn_params_3

    def build_rbm(v, W, bv, bh, k):
        def gibbs_step(v, binomial=False):
            mean_h = sigm(T.dot(v, W) + bh)
            h = rng.binomial(size=mean_h.shape, n=1, p=mean_h, dtype=dtype)
            mean_v = sigm(T.dot(h, W.T) + bv)
            v = rng.binomial(size=mean_v.shape, n=1, p=mean_v, dtype=theano.config.floatX) if binomial else mean_v
            return mean_v, v

        chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v], n_steps=k)
        v_sample = chain[-1]
        mean_v = gibbs_step(v_sample)[0]
        monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
        monitor = monitor.sum() / v.shape[0]

        def free_energy(v):
            return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()

        cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]
        return v_sample, cost, monitor, updates

    def recurrence(v_t, u1_tm1, u2_tm1, u3_tm1):
        bv_t = bv + T.dot(u3_tm1, Wuv)
        bh_t = bh + T.dot(u3_tm1, Wuh)
        generate = v_t is None

        # generate a probability distribution for the visible units, with certain biases
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t, bh_t, k=15)

        u1_t = build_rnn(rnn_params_1, v_t, u1_tm1)
        u2_t = build_rnn(rnn_params_2, u1_t, u2_tm1)
        u3_t = build_rnn(rnn_params_3, u2_t, u3_tm1)

        return ([v_t, u1_t, u2_t, u3_t], updates) if generate else [u1_t, u2_t, u3_t, bv_t, bh_t]

    v = T.matrix()

    # rnn initial values
    u1_0 = T.zeros((n_hidden_recurrent,))
    u2_0 = T.zeros((n_hidden_recurrent,))
    u3_0 = T.zeros((n_hidden_recurrent,))

    (_, _, _, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u1_tm1, u2_tm1, u3_tm1, *_: recurrence(v_t, u1_tm1, u2_tm1, u3_tm1), sequences=v,
        outputs_info=[u1_0, u2_0, u3_0, None, None])

    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t, bh_t, k=20)
    updates_train.update(updates_rbm)

    (v_t, _, _, _), updates_generate = theano.scan(
        lambda u1_tm1, u2_tm1, u3_tm1, *_: recurrence(None, u1_tm1, u2_tm1, u3_tm1),
        outputs_info=[None, u1_0, u2_0, u3_0], n_steps=20)

    # l1 and l2 regularizers
    for param in rnn_params_1 + rnn_params_2 + rnn_params_3:
        cost += T.sum(param ** 2) * l2_norm * lr
        cost += T.sum(abs(param)) * l1_norm * lr

    return (v, v_sample, cost, monitor, params, updates_train, v_t, updates_generate)


class RnnRbm:
    def __init__(self, n_visible, n_hidden=150, n_hidden_recurrent=100, lr=0.001, l2_norm=0.0001, l1_norm=0.0001):
        (v, v_sample, cost, monitor, params,
         updates_train, v_t, updates_generate) = build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent, lr,
                                                              l2_norm=l2_norm, l1_norm=l1_norm)
        for param in params:
            gradient = T.grad(cost, param, consider_constant=[v_sample])

            # remove nan and inf values
            not_finite = T.or_(T.isnan(gradient), T.isinf(gradient))
            gradient = T.switch(not_finite, 0.1 * param, gradient)
            # max_grad = param * 1e-3
            # gradient = T.switch(T.gt(gradient, max_grad), max_grad, gradient)

            # momentum
            # velocity = shared_zeros('velocity_' + str(param.name), param.get_value(borrow=True).shape)
            # update = param - T.cast(lr, dtype=dtype) * gradient
            # x = momentum * velocity + update - param
            # updates_train[velocity] = x
            # updates_train[param] = momentum * x + update

            # rmsprop
            accu = shared_zeros('accu_' + str(param.name), param.get_value(borrow=True).shape)
            accu_new = 0.9 * accu + 0.1 * gradient ** 2
            updates_train[accu] = accu_new
            updates_train[param] = param - (lr * gradient / T.sqrt(accu_new + 1e-6))
        self.params = params
        self.train_function = theano.function([v], monitor, updates=updates_train)
        self.generate_function = theano.function([], v_t, updates=updates_generate)

    def save(self, dir_path):
        for param in self.params:
            f_name = os.path.join(dir_path, param.name)
            np.save(f_name, param.get_value(borrow=True))

    def load(self, dir_path):
        for param in self.params:
            f_name = os.path.join(dir_path, '%s.npy' % param.name)
            if not os.path.exists(f_name):
                print('File does not exist: "%s" Ignoring...' % f_name)
            else:
                param.set_value(np.load(f_name))


class DirectoryIterator:
    def __init__(self, dir_name):
        assert os.path.isdir(dir_name), 'Invalid directory: "%s"' % dir_name
        self.dir_name = dir_name

    def get_next(self, n_items):
        arrays = itertools.islice(self, n_items)
        return [x for a in arrays for x in a.transpose()]

    def __iter__(self):
        while True:
            dirs = os.listdir(self.dir_name)
            random.shuffle(dirs)
            for f_name in dirs:
                if not f_name.lower().endswith('.npz'):
                    continue
                array = np.load(f_name)
                if len(array) == 0:
                    continue
                yield array / np.max(array)


def train():
    assert 'VOCALIZATION_FILES' in os.environ, 'Must set "VOCALIZATION_FILES" environment variable'
    spec_dir = os.path.join(os.environ['VOCALIZATION_FILES'], 'spectrograms')
    assert os.path.exists(spec_dir), 'Directory "%s" not found: run parser.py' % spec_dir
    os.chdir(spec_dir)

    import pickle as pkl
    info = pkl.load(open('info.pkl', 'rb'))

    save_dir = os.path.join(os.environ['VOCALIZATION_FILES'], 'model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_hidden = int(info['FREQ_DIM'] * hidden_scalar)
    n_hidden_recurrent = int(info['FREQ_DIM'] * hidden_recurrent_scalar)
    for lr in [3e-4, 1e-4, 5e-5, 3e-5, 1e-5]:
        rnnrbm = RnnRbm(n_visible=info['FREQ_DIM'], n_hidden=n_hidden, n_hidden_recurrent=n_hidden_recurrent, lr=lr)
        iterator = DirectoryIterator(spec_dir)

        num_epochs = 50
        batch_size = 20
        rnnrbm.load(save_dir)
        for epoch in range(num_epochs):
            start = time.time()
            costs = list()
            for i in range(0, info['NUM_FILES'], batch_size):
                cost = rnnrbm.train_function(iterator.get_next(batch_size))
                costs.append(cost)
                print('\rEpoch %d (%d / %d) Cost: %.3f' % (epoch, i, info['NUM_FILES'], cost), end='')
            end = time.time()
            print('\rEpoch %d Time: %dm %ds Cost: %.3f' % (epoch,
                                                           int(end - start) / 60,
                                                           int(end - start) % 60,
                                                           np.mean(costs)))
            if (epoch + 1) % 10 == 0:
                figure_dir = os.path.join(os.environ['VOCALIZATION_FILES'], 'figures')
                if not os.path.exists(figure_dir):
                    os.makedirs(figure_dir)
                figure = rnnrbm.generate_function()
                np.save(os.path.join(figure_dir, 'sample_at_epoch_%d_lr_%f.npy' % (epoch + 1, lr)), figure)
                rnnrbm.save(save_dir)


def test():
    assert 'VOCALIZATION_FILES' in os.environ, 'Must set "VOCALIZATION_FILES" environment variable'
    spec_dir = os.path.join(os.environ['VOCALIZATION_FILES'], 'spectrograms')
    assert os.path.exists(spec_dir), 'Directory "%s" not found: run parser.py' % spec_dir
    os.chdir(spec_dir)

    import pickle as pkl
    info = pkl.load(open('info.pkl', 'rb'))

    model_dir = os.path.join(os.environ['VOCALIZATION_FILES'], 'model')
    assert os.path.exists(model_dir), 'Directory "%s" not found: Run training method' % model_dir

    save_dir = os.path.join(os.environ['VOCALIZATION_FILES'], 'generated')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    n_hidden = int(info['FREQ_DIM'] * hidden_scalar)
    n_hidden_recurrent = int(info['FREQ_DIM'] * hidden_recurrent_scalar)
    lr = 1e-4
    rnnrbm = RnnRbm(n_visible=info['FREQ_DIM'], n_hidden=n_hidden, n_hidden_recurrent=n_hidden_recurrent, lr=lr)
    rnnrbm.load(model_dir)

    n_samples = 5
    for sample in range(n_samples):
        figure = rnnrbm.generate_function()
        np.save('sample_%d.npy' % sample, figure)
        pylab.figure()
        pylab.imshow(figure.T, origin='lower', aspect='auto', interpolation='nearest')
    pylab.show()

    print('Done')


if __name__ == '__main__':
    train()
    # test()
