# -*- coding: utf-8 -*-

from base_layer import BaseLayer as BL
import csv
import time
import logger
import numpy as np
from logger import logger

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class MultiBaseLayer(object):

    def __init__(self, **kwargs):

        self.BaseLayer = kwargs.get("base_layer") or BL

        np_seed = kwargs.get("np_seed") or int(time.time())
        logger.info("setting np_seed %r", np_seed)
        self.np_rng = np.random.RandomState(np_seed)

        theano_seed = kwargs.get("theano_seed") or self.np_rng.randint(2 ** 30)
        logger.info("setting theano_seed %r", theano_seed)
        self.theano_rng = RandomStreams(theano_seed)

        file_name = kwargs.get("file_name")
        if file_name:
            self._load_from_file(**kwargs)
        else:
            self._load_from_params(**kwargs)

    def _load_from_file(self, **kwargs):

        file_name = kwargs.get("file_name")
        n_layers = None
        file_list = []
        with open(file_name, "rb") as f:
            reader = csv.reader(f)
            for l in reader:
                if n_layers is None:
                    n_layers = [int(item) for item in l]
                else:
                    file_list.append(l[0])
        assert len(file_list) + 1 == len(n_layers)
        self.n_layers = n_layers
        self.base_layer = []
        self.params = []
        for i in range(0, len(file_list)):
            item_file_name = file_list[i]
            logger.info("loading file %r", item_file_name)
            layer = self.BaseLayer(file_name=item_file_name)
            self.base_layer.append(layer)
            self.params += layer.params

    def _load_from_params(self, **kwargs):

        n_layers = kwargs.get("n_layers")
        assert isinstance(n_layers, (list, tuple))
        self.n_layers = n_layers

        self.base_layer = []
        self.params = []
        for i in range(1, len(n_layers)):
            layer = self.BaseLayer(n_v=n_layers[i - 1], n_h=n_layers[i])
            self.base_layer.append(layer)
            self.params += layer.params

    def param_output(self, **kwargs):

        file_name = kwargs.get("file_name")
        with open(file_name, "wb") as f:
            writer = csv.writer(f)
            writer.writerow(self.n_layers)
            for i in range(0, len(self.base_layer)):
                self.base_layer[i].param_output(file_name="%s_%d" % (file_name, i))
                writer.writerow(("%s_%d" % (file_name, i),))

    def get_output_by_input(self, **kwargs):

        input_val = kwargs.get("input_val") or []
        tmp_val = input_val
        for layer in self.base_layer:
            tmp_val = layer.get_output_by_input(input_val=tmp_val)
        return tmp_val


if __name__ == "__main__":
    multi_base_layer = MultiBaseLayer(file_name="test_multi.csv")
    for layer in multi_base_layer.base_layer:
        print layer.W.get_value(borrow=True)
    multi_base_layer.param_output(file_name="test_multi.csv")