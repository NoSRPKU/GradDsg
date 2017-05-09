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
from common import get_list_from_val

class DeepNet(MultiBaseLayer):

    def get_cost(self, **kwargs):

        input_val = kwargs.get("input_val_cost", [])
        output_val = kwargs.get("output_val_cost", [])

        net_output = self.get_output_by_input(input_val=input_val)
        cost = T.mean(T.sum(T.abs_(output_val - net_output), axis=1))#T.mean(- T.sum(net_output * T.log(output_val) + (1 - net_output) * T.log(1 - output_val), axis=1))
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

    def get_output_by_input_with_fil(self, **kwargs):

        input_val = kwargs.get("input_val") or []
        tmp_val = input_val
        add_val = None
        for layer in self.base_layer:
            tmp_val, tmp_add_val = layer.get_output_by_input_with_fil(input_val=tmp_val)
            add_val = np.c_[tmp_add_val, add_val] if add_val is not None else tmp_add_val
        return tmp_val, add_val

    def pretrain(self, **kwargs):

        input_v = kwargs.get("input_val", [])
        for layer in self.base_layer:
            layer.autoencoder_pretrain(input_val=input_v, rand_epoch=500, epoch=1, sample_size=len(train_inp),
                                       batch_size=1000, pre_cease_threshold=0.00001)
            input_v = get_val(layer.get_output_by_input(input_val=input_v))
            print type(input_v), input_v
            input_v = input_v.tolist()


    def train(self, **kwargs):
        #self.pretrain(**kwargs)

        input_v = kwargs.get("input_val", [])
        output_v = kwargs.get("output_val", [])

        if isinstance(input_v, (list, tuple)):
            input_v = np.asarray(input_v, dtype=DTYPE)
        if isinstance(input_v, np.ndarray):
            input_v = theano.shared(value=input_v, name="input_val")

        tmp_ov = []
        if isinstance(output_v, (list, tuple)):
            output_v = np.asarray(output_v, dtype=DTYPE)
            output_v = self.activation(output_v)
            tmp_ov = get_val(output_v)
        if isinstance(output_v, np.ndarray):
            output_v = theano.shared(value=output_v, name="output_val")

        lr = kwargs.get("learning_rate") or 0.2

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
        return tmp_ov

if __name__ == '__main__':
    def actt(x):
        return x * x
    from generate_data.generate_data import GenerateData
    data = GenerateData()#ßfile_name="data/inputfile.csv")
    train_inp, train_outp = data.get_train_data()
    verify_inp, verify_outp = data.get_verify_data()
    data.save_to_file("data/inputfile.csv")
    net = DeepNet(base_layer=ReverseLayer, n_layers=[30, 30, 25, 16, 9, 1], activation=T.nnet.softplus)
    ov = net.train(input_val=train_inp, output_val=train_outp, rand_epoch=3000, epoch=1000, sample_size=len(train_inp), batch_size=1000, pre_cease_threshold=0.00001)
    net.param_output(file_name="data/train/train_0003.csv")

    out1, out2 = net.get_output_by_input_with_fil(input_val=verify_inp)
    outt = net.get_output_by_input(input_val=verify_inp)

    for layer in net.base_layer:
        print get_val(layer.activation([-1, -0.5, 0, 0.5, 1]))
    print get_val(net.activation([-1, -0.5, 0, 0.5, 1]))

    import csv

    with open("test_out_k002.csv", "wb") as f:
        writer = csv.writer(f)
        for line in zip(out1.tolist(), get_val(net.activation(verify_outp)).tolist(), out2.tolist(), verify_inp, get_val(outt).tolist(), verify_outp):
            writer.writerow(get_list_from_val(line))

if __name__ == "！！__main__":
    #data = LoadTrafficDataForLearning(path="/home/nosr/Documents/out", file_name="AL1644.csv")
    #data.generate_data(out_column=[0])
    #train_inp, train_outp = data.get_train_data()
    verify_inp, verify_outp = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9]], [[0]]
    #verify_inp, verify_outp = data.get_verify_data()
    #test_inp, test_outp = data.get_test_data()

    net = DeepNet(base_layer=ReverseLayer, file_name="test_multi.csv")
    #net = DeepNet(n_layers=[16, 32, 12, 6, 1])
    #net.train(input_val=verify_inp, output_val=verify_outp, rand_epoch=1000, epoch=1000, sample_size=len(verify_inp), batch_size=2000)
    #net.param_output(file_name="test_multi_haha.csv")
    out1, out2 =  net.get_output_by_input_with_fil(input_val=verify_inp)
    #out3, out4 = net.get_output_by_input_with_fil(input_val=train_inp)
    #out5, out6 = net.get_output_by_input_with_fil(input_val=test_inp)

    print out2

    import csv
    with open("test_out.csv", "wb") as f:
        writer = csv.writer(f)
        for line in zip(out1.tolist(), get_val(T.nnet.sigmoid(verify_outp)).tolist(), out2.tolist()):
            writer.writerow(get_list_from_val(line))
    with open("test_out2.csv", "wb") as f:
        writer = csv.writer(f)
        for line in zip(out3.tolist(), get_val(T.nnet.sigmoid(train_outp)).tolist(), out4.tolist()):
            writer.writerow(get_list_from_val(line))
    with open("test_out3.csv", "wb") as f:
        writer = csv.writer(f)
        for line in zip(out5.tolist(), get_val(T.nnet.sigmoid(test_outp)).tolist(), out6.tolist()):
            writer.writerow(get_list_from_val(line))


    for layer in net.base_layer:
        print layer.W_fil
        print layer.W_res
