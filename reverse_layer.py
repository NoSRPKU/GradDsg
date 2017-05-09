# -*- coding: utf-8 -*-

from utils.base_layer import BaseLayer

import numpy as np
from scipy import linalg
import theano
from theano import tensor as T
from logger import logger
from utils import get_val
from common import get_list_from_val

class ReverseLayer(BaseLayer):

    def __init__(self, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.W_fil = None
        self.b_h_fil = None

    def get_svd(self):

        W = self.W.get_value(borrow=True)
        U, s, Vh = linalg.svd(W)
        return U, s, Vh, W

    def get_fil_weight(self):

        U, s, Vh, W = self.get_svd()
        if self.n_v <= self.n_h:
            self.W_fil = self.W
            self.b_h_fil = self.b_h
            return
        S1 = np.append(s, np.ones(self.n_v - self.n_h))
        S2 = np.diag(S1)
        Vh1 = np.c_[Vh, np.zeros(shape=(self.n_h, self.n_v - self.n_h))]
        Vh2 = np.c_[np.zeros(shape=(self.n_v - self.n_h, self.n_h)), np.eye(self.n_v - self.n_h)]
        Vh3 = np.r_[Vh1, Vh2]
        self.W_fil = np.dot(U, np.dot(S2, Vh3))
        self.b_h_fil = np.append(self.b_h.get_value(borrow=True), np.zeros(self.n_v - self.n_h))
        logger.info("W_fil shape %s", self.W_fil.shape)
        logger.info("b_h_fil shape %s, %s %s", self.b_h_fil.shape, self.b_h_fil, self.b_h)

        _S1 = 1 / S1
        _S2 = np.diag(_S1)

        self.W_res = np.dot(Vh3.T, np.dot(_S2, U.T))

    def get_output_by_input_with_fil(self, **kwargs):

        if self.W_fil is None:
            self.get_fil_weight()

        input_val = kwargs.get("input_val", [])
        tmp_output = T.dot(input_val, self.W_fil) + self.b_h_fil
        nxt_val, pp_val = tmp_output[:, 0:self.n_h], tmp_output[:, self.n_h:self.n_v]
        output_val = self.activation(nxt_val)
        output_val = get_val(output_val)
        pp_val = get_val(pp_val)
        logger.info("fil shape %s", output_val.shape)
        return output_val, pp_val

if __name__ == "__main__":

    from utils.multi_base_layer import MultiBaseLayer
    multi_layer = MultiBaseLayer(base_layer=ReverseLayer, file_name="test_multi.csv")
    for layer in multi_layer.base_layer:
        U, s, Vh, W = layer.get_svd()
        print U.shape, s.shape, Vh.shape, W.shape
        if layer.n_h < layer.n_v:
            S1 = np.append(s, np.ones(layer.n_v - layer.n_h))
            S2 = np.diag(S1)
            #print S2
            Vh1 = np.c_[Vh, np.zeros(shape=(layer.n_h, layer.n_v - layer.n_h))]
            Vh2 = np.c_[np.zeros(shape=(layer.n_v - layer.n_h, layer.n_h)), np.eye(layer.n_v - layer.n_h)]
            #print Vh1.shape, Vh2.shape
            Vh3 = np.r_[Vh1, Vh2]
            #print Vh3
            Res = np.dot(U, np.dot(S2, Vh3))
            ext_W = np.c_[W, np.zeros(shape=(layer.n_v, layer.n_v - layer.n_h))]

            _S1 = 1 / S1
            _S2 = np.diag(_S1)

            _Res = np.dot(Vh3.T, np.dot(_S2, U.T))