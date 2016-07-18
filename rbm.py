from __future__ import print_function

import itertools

import pylab
import scipy
import theano
import theano.tensor as T
import numpy as np
import os
import time

from scipy.io.wavfile import write as wavwrite
from utils import sigm, dtype, rng, shared_normal, shared_zeros, relu


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


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):

    # rbm params
    W = shared_normal('W', n_visible, n_hidden, scale=0.01)
    bv = shared_zeros('bv', n_visible)
    bh = shared_zeros('bh', n_hidden)

    # rnn -> rbm connections
    Wuh = shared_normal('Wuh', n_hidden_recurrent, n_hidden, scale=0.0001)
    Wuv = shared_normal('Wuv', n_hidden_recurrent, n_visible, scale=0.0001)

    params = [W, bv, bh, Wuh, Wuv]

    # update gate
    w_in_update = shared_normal('w_in_update', n_visible, n_hidden_recurrent, scale=0.0001)
    w_hidden_update = shared_normal('w_hidden_update', n_hidden_recurrent, n_hidden_recurrent, scale=0.0001)
    b_update = shared_zeros('b_update', n_hidden_recurrent)
    params += [w_in_update, w_hidden_update, b_update]

    # reset gate
    w_in_reset = shared_normal('w_in_reset', n_visible, n_hidden_recurrent, scale=0.0001)
    w_hidden_reset = shared_normal('w_hidden_reset', n_hidden_recurrent, n_hidden_recurrent, scale=0.0001)
    b_reset = shared_zeros('b_reset', n_hidden_recurrent)
    params += [w_in_reset, w_hidden_reset, b_reset]

    # hidden layer
    w_in_hidden = shared_normal('w_in_hidden', n_visible, n_hidden_recurrent, scale=0.0001)
    w_reset_hidden = shared_normal('w_reset_hidden', n_hidden_recurrent, n_hidden_recurrent, scale=0.0001)
    b_hidden = shared_zeros('b_hidden', n_hidden_recurrent)
    params += [w_in_hidden, w_reset_hidden, b_hidden]

    v = T.matrix()
    u0 = T.zeros((n_hidden_recurrent,)) # rnn initial value

    def recurrence(v_t, u_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh_t = bh + T.dot(u_tm1, Wuh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t, bh_t, k=25)

        # gru equations
        update_gate = relu(T.dot(v_t, w_in_update) + T.dot(u_tm1, w_hidden_update) + b_update)
        reset_gate = relu(T.dot(v_t, w_in_reset) + T.dot(u_tm1, w_hidden_reset) + b_reset)
        u_t_temp = T.tanh(T.dot(v_t, w_in_hidden) + T.dot(u_tm1 * reset_gate, w_reset_hidden) + b_hidden)
        u_t = (1 - update_gate) * u_t_temp + update_gate * u_tm1

        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    (u_t, bv_t, bh_t), updates_train = theano.scan(lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1), sequences=v,
                                                   outputs_info=[u0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:], k=15)
    updates_train.update(updates_rbm)

    (v_t, u_t), updates_generate = theano.scan(lambda u_tm1, *_: recurrence(None, u_tm1), outputs_info=[None, u0],
                                               non_sequences=params, n_steps=200)
    return (v, v_sample, cost, monitor, params, updates_train, v_t, updates_generate)


class RnnRbm:
    def __init__(self, n_visible, n_hidden=150, n_hidden_recurrent=100, lr=0.001, momentum=0.5):
        (v, v_sample, cost, monitor,
         params, updates_train, v_t, updates_generate) = build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent)
        for param in params:
            gradient = T.grad(cost, param, consider_constant=[v_sample])

            # clipping
            not_finite = T.or_(T.isnan(gradient), T.isinf(gradient))
            gradient = T.switch(not_finite, 0.1 * param, gradient)
            # max_grad = param * 1e-3
            # gradient = T.switch(T.gt(gradient, max_grad), max_grad, gradient)

            velocity = shared_zeros('velocity_' + str(param.name), param.get_value(borrow=True).shape)
            update = param - T.cast(lr, dtype=dtype) * gradient
            x = momentum * velocity + update - param
            updates_train[velocity] = x
            updates_train[param] = momentum * x + update
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
            for f_name in os.listdir(self.dir_name):
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

    n_hidden = int(info['FREQ_DIM'] * 0.8)
    n_hidden_recurrent = int(info['FREQ_DIM'] * 0.7)
    lr = 1e-4
    rnnrbm = RnnRbm(n_visible=info['FREQ_DIM'], n_hidden=n_hidden, n_hidden_recurrent=n_hidden_recurrent, lr=lr)
    iterator = DirectoryIterator(spec_dir)

    num_epochs = 20
    batch_size = 20
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

    n_hidden = int(info['FREQ_DIM'] * 0.8)
    n_hidden_recurrent = int(info['FREQ_DIM'] * 0.7)
    lr = 1e-4
    rnnrbm = RnnRbm(n_visible=info['FREQ_DIM'], n_hidden=n_hidden, n_hidden_recurrent=n_hidden_recurrent, lr=lr)
    rnnrbm.load(model_dir)

    n_samples = 6
    for sample in range(n_samples):
        figure = rnnrbm.generate_function()
        np.save('sample_%d.npy' % sample, figure)
        pylab.figure()
        pylab.imshow(figure.T, origin='lower', aspect='auto', interpolation='nearest')
    pylab.show()

if __name__ == '__main__':
    train()
    # test()