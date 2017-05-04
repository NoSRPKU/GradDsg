# -*- coding: utf-8 -*-

import theano
import numpy as np
import theano.tensor as T
import time
from logger import logger
from common import DTYPE
import csv

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class BaseLayer(object):

    def __init__(self, **kwargs):

        np_seed = kwargs.get("np_seed") or int(time.time())
        logger.info("setting np_seed %r", np_seed)
        self.np_rng = np.random.RandomState(np_seed)

        theano_seed = kwargs.get("theano_seed") or self.np_rng.randint(2 ** 30)
        logger.info("setting theano_seed %r", theano_seed)
        self.theano_rng = RandomStreams(theano_seed)

        input_file = kwargs.get("file_name")
        if input_file:
            self._load_from_file(**kwargs)
        else:
            self._load_from_params(**kwargs)
        self.params = [self.W, self.b_h]

    def _load_from_params(self, **kwargs):

        n_v = kwargs.get("n_v")
        assert isinstance(n_v, (int, long))
        assert n_v > 0
        self.n_v = n_v
        logger.info("setting n_v %r", self.n_v)

        n_h = kwargs.get("n_h")
        assert isinstance(n_h, (int, long))
        assert n_h > 0
        self.n_h = n_h
        logger.info("setting n_h %r", self.n_h)

        init_W = kwargs.get("init_W")
        self.W = self._get_W_by_init(W=init_W, name="W")

        init_b_h = kwargs.get("init_b_h")
        self.b_h = self._get_b_by_init(b=init_b_h, n=self.n_h, name="b_h")

    def _load_from_file(self, **kwargs):

        file_name = kwargs.get("file_name")
        n_h = None
        n_v = None
        init_W = []
        init_b_h = None
        with open(file_name, "rb") as f:
            reader = csv.reader(f)
            for l in reader:
                if n_h is None and n_v is None:
                    n_h, n_v = [int(item) for item in l]
                elif len(init_W) < n_v:
                    init_W.append([float(item) for item in l])
                elif init_b_h is None:
                    init_b_h = [float(item) for item in l]

        self.n_h = n_h
        self.n_v = n_v
        self.W = self._get_W_by_init(W=init_W, n_h=self.n_h, n_v=self.n_v, name="W")
        self.b_h = self._get_b_by_init(b=init_b_h, n=self.n_h, name="b_h")

    def param_output(self, **kwargs):

        file_name = kwargs.get("file_name")
        with open(file_name, "wb") as f:
            writer = csv.writer(f)
            writer.writerow((self.n_h, self.n_v))
            writer.writerows(self.W.get_value(borrow=True).tolist())
            writer.writerow(self.b_h.get_value(borrow=True).tolist())

    def _get_W_by_init(self, **kwargs):

        W = kwargs.get("W")
        n_v = kwargs.get("n_v") or self.n_v
        n_h = kwargs.get("n_h") or self.n_h
        name = kwargs.get("name") or "W"
        if W is None:
            init_W_limit = kwargs.get("init_W_limit")
            np_rng = kwargs.get("np_rng") or self.np_rng
            if init_W_limit is None:
                init_W_limit = 4. * np.sqrt((6. / (n_v + n_h)))
            logger.info("setting init_W_limit for %r %r", name, init_W_limit)
            return theano.shared(value=np.asarray(np_rng.uniform(low=-init_W_limit, high=init_W_limit,
                                                      size=(n_v, n_h)), dtype=DTYPE),
                                 name=name, borrow=True)
        elif isinstance(W, (list, tuple, np.ndarray)):
            if isinstance(W, (list, tuple)):
                W = np.asarray(W, dtype=DTYPE)
            n_v_W, n_h_W = W.shape
            assert n_v_W == n_v
            assert n_h_W == n_h
            return theano.shared(value=W, name=name, borrow=True)
        elif isinstance(W, theano.Variable):
            return W
        else:
            raise AttributeError("init W for %r type %r error", name, type(W))

    def _get_b_by_init(self, **kwargs):

        b = kwargs.get("b")
        n = kwargs.get("n")
        name = kwargs.get("name") or "bias"

        if b is None:
            return theano.shared(value=np.zeros(shape=n, dtype=DTYPE), name=name, borrow=True)
        elif isinstance(b, (list, tuple, np.ndarray)):
            if isinstance(b, (list, tuple)):
                b = np.asarray(b, dtype=DTYPE)
            assert b.shape[0] == n
            return theano.shared(value=b, name=name, borrow=True)
        elif isinstance(b, theano.Variable):
            return b
        else:
            raise AttributeError("init b for %r type %r error", name, type(b))

    def get_output_by_input(self, **kwargs):

        input_val = kwargs.get("input_val") or []
        output_val = T.nnet.sigmoid(T.dot(input_val, self.W) + self.b_h)
        return output_val

if __name__ == "__main__":
    base_layer = BaseLayer(n_v=2, n_h=2)

    print base_layer.W.get_value(borrow=True)
    print base_layer.b_h.get_value(borrow=True)