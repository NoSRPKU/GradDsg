# -*- coding: utf-8 -*-

import csv
from logger import logger
from common import get_list_from_val, DTYPE

import numpy as np

class LoadData(object):

    def __init__(self, path, file_name):
        self.path = path
        if '/' != self.path[-1]:
            self.path += '/'
        self.file_name = file_name
        self.file_route = self.path + self.file_name
        logger.info("opening file:%s", self.file_route)
        raw_data_from_file = []
        self.column_count = 0
        with open(self.file_route, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                parsed_row = []
                for r in row:
                    try:
                        r = float(r)
                    except:
                        pass
                    parsed_row.append(r)
                raw_data_from_file.append(tuple(parsed_row))
                if not self.column_count:
                    self.column_count = len(row)
                    assert self.column_count > 0

                assert len(row) == self.column_count
        self.raw_data = tuple(raw_data_from_file)

        self.line_count = len(self.raw_data)
        logger.info("line count:%d", self.line_count)
        logger.info("column count:%d", self.column_count)

class LoadDataWithRegularizeParam(LoadData):

    def __init__(self, path, file_name, column=None):
        super(LoadDataWithRegularizeParam, self).__init__(path, file_name)

        if not column:
            self.column = tuple(range(0, self.column_count))
        else:
            self.column = get_list_from_val(column)
        logger.info("column selected:%s", self.column)

        for col in self.column:
            assert col < self.column_count
            assert col >= 0

        self.selected_data = tuple([tuple([val[i] for i in self.column]) for val in self.raw_data])
        np_data = np.asarray(self.selected_data, dtype=DTYPE)
        self.average = tuple(np.average(np_data, axis=0).tolist())
        self.std = tuple(np.std(np_data, axis=0).tolist())
        logger.debug("average:%s", self.average)
        logger.debug("std:%s", self.std)

if __name__ == "__main__":
    data = LoadDataWithRegularizeParam(path="/home/nosr/Documents/out", file_name="AL2949.csv", column=range(4, 9))