# -*- coding: utf-8 -*-

from utils.multi_base_layer import MultiBaseLayer

from theano import tensor as T
import numpy as np

class DeepNet(MultiBaseLayer):

    def __init__(self, **kwargs):
        super(DeepNet, self).__init__(**kwargs)

    