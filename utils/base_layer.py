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

        activation = kwargs.get("activation")
        if activation is None:
            logger.info("base layer activation sigmoid")
            self.activation = T.nnet.sigmoid
        else:
            self.activation = activation

        self.Wt = self._get_W_by_init(n_v=self.n_h, n_h=self.n_v, name="Wt")
        self.b_v = self._get_b_by_init(n = self.n_v, name="b_v")

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
                init_W_limit = 1. * np.sqrt((6. / (n_v + n_h)))
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
        output_val = self.activation(T.dot(input_val, self.W) + self.b_h)
        return output_val

    def get_recon_by_input(self, **kwargs):

        input_val = kwargs.get("input_val") or []
        output_val = self.get_output_by_input(input_val=input_val)
        recon_val = self.activation(T.dot(output_val, self.Wt) + self.b_v)
        return recon_val

    def get_cost(self, **kwargs):

        input_val = kwargs.get("input_val_cost", [])
        output_val = kwargs.get("output_val_cost", [])

        net_output = self.get_recon_by_input(input_val=input_val)
        cost = T.mean(T.sum((output_val - net_output) ** 2, axis=1))#T.mean(- T.sum(net_output * T.log(output_val) + (1 - net_output) * T.log(1 - output_val), axis=1))
        return cost


    def get_updates(self, **kwargs):

        learning_rate = kwargs.get("learning_rate") or 0.1
        params = kwargs.get("params") or self.params
        cost = self.get_cost(**kwargs)
        gparams = T.grad(cost, params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]
        return cost, updates

    def get_training_batch_function(self, **kwargs):

        input_v = kwargs.get("input_val", [])
        output_v = kwargs.get("output_val", [])

        input_val = T.matrix('input', dtype=DTYPE)
        output_val = T.matrix('output', dtype=DTYPE)
        begin = T.iscalar()
        end = T.iscalar()

        cost, updates = self.get_updates(input_val_cost=input_val, output_val_cost=output_val, **kwargs)
        training_batch = theano.function(inputs=[begin, end], outputs=cost, updates=updates,
                                         givens={input_val:input_v[begin:end], output_val:output_v[begin:end]})
        return training_batch

    def autoencoder_pretrain(self, **kwargs):

        input_v = kwargs.get("input_val", [])
        output_v = input_v

        params = [self.W, self.Wt, self.b_v, self.b_h]

        if isinstance(input_v, (list, tuple)):
            input_v = np.asarray(input_v, dtype=DTYPE)
        if isinstance(input_v, np.ndarray):
            input_v = theano.shared(value=input_v, name="input_val")

        if isinstance(output_v, (list, tuple)):
            output_v = np.asarray(output_v, dtype=DTYPE)
            output_v = self.activation(output_v)
        if isinstance(output_v, np.ndarray):
            output_v = theano.shared(value=output_v, name="output_val")

        lr = kwargs.get("learning_rate") or 0.2

        train_batch = self.get_training_batch_function(input_val=input_v, output_val=output_v, learning_rate=lr, params=params)

        rand_epoch = kwargs.get("rand_epoch") or 10000
        epoch = kwargs.get("epoch") or 100
        batch_size = kwargs.get("batch_size") or 1000
        sample_size = kwargs.get("sample_size") or 0

        pre_cease_threshold = kwargs.get("pre_cease_threshold") or 0.00001

        assert sample_size > 0

        for e in range(0, rand_epoch):
            index = self.np_rng.randint(0, sample_size - batch_size)
            cost = train_batch(index, index + batch_size)
            logger.info("rand pretrain epoch %s %s %s", e, cost, index)

        epoch_avg_cost = None
        avg_cost = {}
        for e in range(0, epoch):
            index = 0
            flag_for_epoch = False
            while index + batch_size <= sample_size:
                cost = train_batch(index, index + batch_size)
                prev_cost = avg_cost.get(index)
                logger.info("pretrain epoch %s %s %s %s %s", e, index, cost, prev_cost, epoch_avg_cost)
                if prev_cost is None or prev_cost - cost > pre_cease_threshold:
                    flag_for_epoch = True
                avg_cost[index] = cost
                index += batch_size
            if not flag_for_epoch:
                break
            new_avg_cost = np.mean(avg_cost.values())
            if epoch_avg_cost is not None and epoch_avg_cost < new_avg_cost:
                break
            epoch_avg_cost = new_avg_cost

if __name__ == "__main__":
    base_layer = BaseLayer(n_v=2, n_h=2)

    print base_layer.W.get_value(borrow=True)
    print base_layer.b_h.get_value(borrow=True)