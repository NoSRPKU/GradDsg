# -*- coding: utf-8 -*-

import theano

DTYPE = theano.config.floatX

def get_list_from_val(val):
    ret_list = []
    if isinstance(val, list):
        for v in val:
            ret_list += get_list_from_val(v)
    elif isinstance(val, tuple):
        for v in val:
            ret_list += get_list_from_val(v)
    elif isinstance(val, dict):
        for v in val.values():
            ret_list += get_list_from_val(v)
    else:
        ret_list = [val]
    return tuple(ret_list)