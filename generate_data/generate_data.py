# -*- coding: utf-8 -*-

import numpy as np
import time
import csv

class GenerateData(object):

    def __init__(self, **kwargs):

        file_name = kwargs.get("file_name")
        if file_name is None:
            data_size = kwargs.get("data_size") or 50000
            verify_size = kwargs.get("verify_size") or 50000
            test_size = kwargs.get("test_size") or 50000
            dimension = kwargs.get("dimension") or 30
            func = kwargs.get("func") or self._default_func
            np_rng = np.random.RandomState(int(time.time()))
            self.data_size = data_size
            self.verify_size = verify_size
            self.test_size = test_size
            self.dimension = dimension

            print np.asarray(range(0, dimension)) * 2. / dimension - 1

            self.raw_input_data = np_rng.uniform(low=-1, high=1, size=(data_size,dimension)) + (np.asarray(range(0, dimension)) * 4. / dimension - 2)
            self.raw_output_data = self._generate_output(self.raw_input_data, func)
            self.raw_input_mean = np.asarray([0])#np.mean(self.raw_input_data, axis=0)
            self.raw_input_std = np.asarray([1])#np.std(self.raw_input_data, axis=0)
            self.raw_output_mean = np.asarray([0])#np.mean(self.raw_output_data, axis=0)
            self.raw_output_std = np.asarray([3])#np.std(self.raw_output_data, axis=0)
            self.train_input_data = tuple(((self.raw_input_data - self.raw_input_mean) / self.raw_input_std).tolist())
            self.train_output_data = tuple(((self.raw_output_data - self.raw_output_mean) / self.raw_output_std).tolist())
            self.raw_verify_input_data = np_rng.uniform(low=-1, high=1, size=(verify_size,dimension)) + (np.asarray(range(0, dimension)) * 4. / dimension - 2)
            self.raw_verify_output_data = self._generate_output(self.raw_verify_input_data, func)
            self.verify_input_data = tuple(((self.raw_verify_input_data - self.raw_input_mean) / self.raw_input_std).tolist())
            self.verify_output_data = tuple(((self.raw_verify_output_data - self.raw_output_mean) / self.raw_output_std).tolist())
            self.raw_test_input_data = np_rng.rand(test_size, dimension)
            self.raw_test_output_data = np_rng.uniform(low=-1, high=1, size=(test_size,dimension)) + (np.asarray(range(0, dimension)) * 4. / dimension - 2)
            self.test_input_data = tuple(((self.raw_test_input_data - self.raw_input_mean) / self.raw_input_std).tolist())
            self.test_output_data = tuple(((self.raw_test_output_data - self.raw_output_mean) / self.raw_output_std).tolist())
        else:
            self._load_from_file(file_name=file_name)

    def _generate_output(self, input_data, func):

        if isinstance(input_data, np.ndarray):
            input_data = input_data.tolist()
        tmp_output = []
        for row in input_data:
            tmp_output.append([func(row)])
        return tmp_output

    def get_train_data(self):
        return self.train_input_data, self.train_output_data

    def get_verify_data(self):
        return self.verify_input_data, self.verify_output_data

    def get_test_data(self):
        return self.test_input_data, self.test_output_data

    def _default_func(self, row):
        param = 1#np.log(np.asarray(range(0, len(row))) + 1) + 2
        if isinstance(row, (list, tuple)):
            row = np.asarray(row)
        return np.sqrt(np.sum((row ** 2) * param))

    def save_to_file(self, file_name):

        with open(file_name, "wb") as f_main:
            main_writer = csv.writer(f_main)
            main_writer.writerow((self.data_size, self.verify_size, self.test_size, self.dimension))
            main_writer.writerow(self.raw_input_mean.tolist())
            main_writer.writerow(self.raw_input_std.tolist())
            main_writer.writerow(self.raw_output_mean.tolist())
            main_writer.writerow(self.raw_output_std.tolist())
            train_input_file_name = file_name + "_train"
            train_output_file_name = file_name +  "_train_out"
            main_writer.writerow((train_input_file_name, train_output_file_name))
            with open(train_input_file_name, "wb") as f_train:
                train_writer = csv.writer(f_train)
                for row in self.train_input_data:
                    train_writer.writerow(row)
            with open(train_output_file_name, "wb") as f_train_out:
                train_out_writer = csv.writer(f_train_out)
                for row in self.train_output_data:
                    train_out_writer.writerow(row)

            verify_input_file_name = file_name + "_verify"
            verify_output_file_name = file_name + "_verify_out"
            main_writer.writerow((verify_input_file_name, verify_output_file_name))
            with open(verify_input_file_name, "wb") as f_verify:
                verify_writer = csv.writer(f_verify)
                for row in self.verify_input_data:
                    verify_writer.writerow(row)
            with open(verify_output_file_name, "wb") as f_verify_out:
                verify_out_writer = csv.writer(f_verify_out)
                for row in self.verify_output_data:
                    verify_out_writer.writerow(row)
                    verify_out_writer.writerow(row)

            test_input_file_name = file_name + "_test"
            test_output_file_name = file_name + "_test_out"
            main_writer.writerow((test_input_file_name, test_output_file_name))
            with open(test_input_file_name, "wb") as f_test:
                test_writer = csv.writer(f_test)
                for row in self.test_input_data:
                    test_writer.writerow(row)
            with open(test_output_file_name, "wb") as f_test_out:
                test_out_writer = csv.writer(f_test_out)
                for row in self.test_output_data:
                    test_out_writer.writerow(row)
                    test_out_writer.writerow(row)

    def _load_from_file(self, file_name):

        with open(file_name, "rb") as f_main:
            main_reader = csv.reader(f_main)
            size = None
            input_mean = None
            input_std = None
            output_mean = None
            output_std = None
            train_file = None
            verify_file = None
            test_file = None
            for row in main_reader:
                if size is None:
                    size = row
                    self.data_size = int(size[0])
                    self.verify_size = int(size[1])
                    self.test_size = int(size[2])
                    self.dimension = int(size[3])
                elif input_mean is None:
                    input_mean = row
                    self.raw_input_mean = np.asarray([float(a) for a in input_mean])
                elif input_std is None:
                    input_std = row
                    self.raw_input_std = np.asarray([float(a) for a in input_std])
                elif output_mean is None:
                    output_mean = row
                    self.raw_output_mean = np.asarray([float(a) for a in output_mean])
                elif output_std is None:
                    output_std = row
                    self.raw_output_std = np.asarray([float(a) for a in output_std])
                elif train_file is None:
                    train_file = row
                elif verify_file is None:
                    verify_file = row
                elif test_file is None:
                    test_file = row

        with open(train_file[0]) as train_f:
            train_reader = csv.reader(train_f)
            self.train_input_data = []
            for row in train_reader:
                self.train_input_data.append(tuple([float(a) for a in row]))
            self.train_input_data = tuple(self.train_input_data)
        with open(train_file[1]) as train_out_f:
            train_out_reader = csv.reader(train_out_f)
            self.train_output_data = []
            for row in train_out_reader:
                self.train_output_data.append(tuple([float(a) for a in row]))
            self.train_output_data = tuple(self.train_output_data)

        with open(verify_file[0]) as verify_f:
            verify_reader = csv.reader(verify_f)
            self.verify_input_data = []
            for row in verify_reader:
                self.verify_input_data.append(tuple([float(a) for a in row]))
            self.verify_input_data = tuple(self.verify_input_data)
        with open(verify_file[1]) as verify_out_f:
            verify_out_reader = csv.reader(verify_out_f)
            self.verify_output_data = []
            for row in verify_out_reader:
                self.verify_output_data.append(tuple([float(a) for a in row]))
            self.verify_output_data = tuple(self.verify_output_data)

        with open(test_file[0]) as test_f:
            test_reader = csv.reader(test_f)
            self.test_input_data = []
            for row in test_reader:
                self.test_input_data.append(tuple([float(a) for a in row]))
            self.test_input_data = tuple(self.test_input_data)
        with open(test_file[1]) as test_out_f:
            test_out_reader = csv.reader(test_out_f)
            self.test_output_data = []
            for row in test_out_reader:
                self.test_output_data.append(tuple([float(a) for a in row]))
            self.test_output_data = tuple(self.test_output_data)

        print self.train_input_data[:1]
        print self.train_output_data[:1]
        print self.verify_input_data[:1]
        print self.verify_output_data[:1]
        print self.test_input_data[:1]
        print self.test_output_data[:1]

if __name__ == "__main__":
    data = GenerateData(file_name="data/test_generate_data.csv")
    data.save_to_file(file_name="data/test_generate_data.csv")