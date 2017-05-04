# -*- coding: utf-8 -*-

from loaddata import LoadDataWithRegularizeParam
import datetime
from logger import logger
from common import get_list_from_val

class LoadTrafficData(object):

    def __init__(self, path, file_name, avg_to=0, std_to=1, select_column=None):
        if not select_column:
            self.select_column = tuple([4, 8])
        else:
            self.select_column = get_list_from_val(select_column)
        self.column_count = len(self.select_column)
        logger.info("start loading data from file")
        self.data = LoadDataWithRegularizeParam(path=path, file_name=file_name, column=self.select_column)
        logger.info("loading data from file finished")
        self.data_dict = {}
        for val in self.data.raw_data:
            dt = datetime.datetime.strptime(val[2], "%Y-%m-%d %H:%M:%S") + datetime.timedelta(minutes=int(val[3]) * 15)
            v = tuple([(val[col] - avg + avg_to) * std_to / std
                       for col, avg, std in zip(self.select_column, self.data.average, self.data.std)])
            self.data_dict[dt] = v
        logger.info("generating data_dict finished")

class LoadTrafficDataForLearning(LoadTrafficData):

    def __init__(self, path, file_name, avg_to=0, std_to=1, select_column=None):
        super(LoadTrafficDataForLearning, self).__init__(path, file_name, avg_to, std_to, select_column)
        self.out_column = tuple()
        self.train_in = tuple()
        self.train_out = tuple()
        self.verify_in = tuple()
        self.verify_out = tuple()
        self.test_in = tuple()
        self.test_out = tuple()

    def generate_data(self, train_beg='2009-01-01 00:00:00', verify_beg='2013-01-01 00:00:00',
                      test_beg='2014-01-01 00:00:00', data_end='2015-01-01 00:00:00', time_interval=8, out_column=None):
        if out_column is None:
            self.out_column = tuple(range(0, self.column_count))
        else:
            self.out_column = get_list_from_val(out_column)
        logger.info("select out column:%s", self.out_column)
        for col in self.out_column:
            assert col >= 0
            assert col < self.column_count
        train_beg_time = datetime.datetime.strptime(train_beg, "%Y-%m-%d %H:%M:%S")
        verify_beg_time = datetime.datetime.strptime(verify_beg, "%Y-%m-%d %H:%M:%S")
        test_beg_time = datetime.datetime.strptime(test_beg, "%Y-%m-%d %H:%M:%S")
        data_end_time = datetime.datetime.strptime(data_end, "%Y-%m-%d %H:%M:%S")
        logger.info("getting train data, begin=%s, end=%s", train_beg, verify_beg)
        self.train_in, self.train_out = self._generate_data_group(train_beg_time, verify_beg_time, time_interval=time_interval)
        logger.info("getting verify data, begin=%s, end=%s", verify_beg, test_beg)
        self.verify_in, self.verify_out = self._generate_data_group(verify_beg_time, test_beg_time, time_interval=time_interval)
        logger.info("getting test data, begin=%s, end=%s", test_beg, data_end)
        self.test_in, self.test_out = self._generate_data_group(test_beg_time, data_end_time, time_interval=time_interval)
        logger.info(("getting data finished"))

    def _get_in_out(self, start_time, time_interval):
        inp = []
        end_time = start_time + datetime.timedelta(minutes=time_interval * 15)
        t = start_time
        while t < end_time:
            inp += self.data_dict.get(t, [])
            t += datetime.timedelta(minutes=15)
        outp_tmp = self.data_dict.get(end_time, [])
        outp = [outp_tmp[col] for col in self.out_column]
        return tuple(inp), tuple(outp)

    def _generate_data_group(self, beg, end, time_interval):
        ret_in = []
        ret_out = []
        t = beg
        end = end - datetime.timedelta(minutes=time_interval * 15)
        while t < end:
            tmp_in, tmp_out = self._get_in_out(t, time_interval)
            ret_in.append(tmp_in)
            ret_out.append(tmp_out)
            t += datetime.timedelta(minutes=15)
        return tuple(ret_in), tuple(ret_out)

    def get_train_data(self):
        return self.train_in, self.train_out

    def get_verify_data(self):
        return self.verify_in, self.verify_out

    def get_test_data(self):
        return self.test_in, self.test_out

if __name__ == "__main__":
    data = LoadTrafficDataForLearning(path="/home/nosr/Documents/out", file_name="AL2949.csv")
    data.generate_data()
    tin, tout = data.get_train_data()