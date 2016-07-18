from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

dtype = theano.config.floatX
sigm = lambda x: T.nnet.sigmoid(x)
np.random.seed(0xaced)
rng = RandomStreams(seed=np.random.randint(1 << 30))


def shared_glorot(name, *shape):
    magnitude = 4 * np.sqrt(6. / sum(shape))
    return theano.shared(name=name, value=np.random.uniform(low=-magnitude, high=magnitude, size=shape).astype(dtype))


def shared_normal(name, *shape, **kwargs):
    return theano.shared(name=name, value=np.random.normal(scale=kwargs.get('scale', 1), size=shape).astype(dtype))


def shared_zeros(name, *shape):
    return theano.shared(name=name, value=np.zeros(*shape, dtype=dtype))
