# -*- coding: utf-8 -*-

import theano

def get_val(val, upd=None):
    f = theano.function([], val, updates=upd)
    return f()
