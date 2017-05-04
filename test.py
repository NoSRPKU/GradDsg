# -*- coding: utf-8 -*-

from loaddata.loaddata_traffic import LoadTrafficDataForLearning
from scipy import linalg

if __name__ == "__main__":

    data = LoadTrafficDataForLearning(path="/home/nosr/Documents/out", file_name="AL1644.csv")
    data.generate_data(out_column=[0])
    train_inp, train_outp = data.get_train_data()

