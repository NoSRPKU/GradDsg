# -*- coding: utf-8 -*-

from utils.multi_base_layer import MultiBaseLayer

import theano
from theano import tensor as T
import numpy as np
from common import DTYPE

from logger import logger

from loaddata.loaddata_traffic import LoadTrafficDataForLearning

import os
import signal
from reverse_layer import ReverseLayer
from utils import get_val

class DeepNet(MultiBaseLayer):

    def get_cost(self, **kwargs):

        input_val = kwargs.get("input_val_cost", [])
        output_val = kwargs.get("output_val_cost", [])

        net_output = self.get_output_by_input(input_val=input_val)
        cost = T.mean(T.sum((net_output - output_val) ** 2, axis=1))
        return cost


    def get_updates(self, **kwargs):

        learning_rate = kwargs.get("learning_rate") or 0.1
        cost = self.get_cost(**kwargs)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
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

    def get_output_by_input_with_fil(self, **kwargs):

        input_val = kwargs.get("input_val") or []
        tmp_val = input_val
        add_val = None
        for layer in self.base_layer:
            tmp_val, tmp_add_val = layer.get_output_by_input_with_fil(input_val=tmp_val)
            add_val = np.c_[tmp_add_val, add_val] if add_val is not None else tmp_add_val
        return tmp_val, add_val

    def train(self, **kwargs):

        input_v = kwargs.get("input_val", [])
        output_v = kwargs.get("output_val", [])

        if isinstance(input_v, (list, tuple)):
            input_v = np.asarray(input_v, dtype=DTYPE)
        if isinstance(input_v, np.ndarray):
            input_v = theano.shared(value=input_v, name="input_val", borrow=True)

        if isinstance(output_v, (list, tuple)):
            output_v = np.asarray(output_v, dtype=DTYPE)
            output_v = T.nnet.sigmoid(output_v)
        if isinstance(output_v, np.ndarray):
            output_v = theano.shared(value=output_v, name="output_val", borrow=True)

        lr = kwargs.get("learning_rate") or 0.1

        train_batch = self.get_training_batch_function(input_val=input_v, output_val=output_v, learning_rate=lr)

        rand_epoch = kwargs.get("rand_epoch") or 10000
        epoch = kwargs.get("epoch") or 100
        batch_size = kwargs.get("batch_size") or 1000
        sample_size = kwargs.get("sample_size") or 0

        pre_cease_threshold = kwargs.get("pre_cease_threshold") or 0.00001

        assert sample_size > 0

        for e in range(0, rand_epoch):
            index = self.np_rng.randint(0, sample_size - batch_size)
            cost = train_batch(index, index + batch_size)
            logger.info("rand train epoch %s %s %s", e, cost, index)

        epoch_avg_cost = None
        avg_cost = {}
        for e in range(0, epoch):
            index = 0
            flag_for_epoch = False
            while index + batch_size <= sample_size:
                cost = train_batch(index, index + batch_size)
                prev_cost = avg_cost.get(index)
                logger.info("train epoch %s %s %s %s %s", e, index, cost, prev_cost, epoch_avg_cost)
                if prev_cost is None or abs(prev_cost - cost) > pre_cease_threshold:
                    flag_for_epoch = True
                avg_cost[index] = cost
                index += batch_size
            if not flag_for_epoch:
                break
            epoch_avg_cost = np.mean(avg_cost.values())

if __name__ == "__main__":
    data = LoadTrafficDataForLearning(path="/home/nosr/Documents/out", file_name="AL1644.csv")
    data.generate_data(out_column=[0])
    train_inp, train_outp = data.get_train_data()
    #verify_inp, verify_outp = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0]]
    verify_inp, verify_outp = data.get_verify_data()
    test_inp, test_outp = data.get_test_data()

    net = DeepNet(base_layer=ReverseLayer, file_name="test_multi.csv")
    #net = DeepNet(n_layers=[16, 32, 12, 6, 1])
    #net.train(input_val=train_inp, output_val=train_outp, rand_epoch=1000, epoch=1000, sample_size=len(train_inp), batch_size=2000)
    #net.param_output(file_name="test_multi_haha.csv")
    out1, out2 =  net.get_output_by_input_with_fil(input_val=verify_inp)
    print np.min(out2, axis=0).tolist()
    out3, out4 = net.get_output_by_input_with_fil(input_val=train_inp)
    print np.min(out4, axis=0).tolist()
    out5, out6 = net.get_output_by_input_with_fil(input_val=test_inp)
    print np.min(out6, axis=0).tolist()

    for layer in net.base_layer:
        print layer.W_fil
        print layer.W_res
