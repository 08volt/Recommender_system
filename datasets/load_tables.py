import time
import warnings
import os
import numpy as np
import scipy.sparse as sps


def rowSplit(rowString):
    split = rowString.split(",")
    split[2] = split[2].replace("\n", "")
    indexes = [0, 0]
    indexes[0] = int(split[0])  # itemID = row
    indexes[1] = int(split[1])  # feature = column
    value = float(split[2])  # value
    return tuple(indexes), value


def load_ICM_csr():
    ICM_file = open_file("data_ICM_title_abstract.csv")
    ICM_file.seek(0)
    ICM_indexes = []
    ICM_values = []
    ICM_file.readline()
    for line in ICM_file:
        i, v = rowSplit(line)
        ICM_indexes.append(i)
        ICM_values.append(v)
    data = np.array(ICM_indexes)
    result = np.zeros((data[:, 0].max() + 1, data[:, 1].max() + 1), dtype=float)
    result[data[:, 0], data[:, 1]] = ICM_values
    ICM_all = sps.coo_matrix(result)
    return ICM_all.tocsr()


def load_URM_csr():
    URM_file = open_file("data_train.csv")
    URM_file.seek(0)
    URM_tuples = []
    URM_file.readline()
    for line in URM_file:
        i, v = rowSplit(line)
        URM_tuples.append([i[0], i[1], int(v)])
    data = np.array(URM_tuples)
    result = np.zeros((data[:, 0].max() + 1, data[:, 1].max() + 1), dtype=int)
    result[data[:, 0], data[:, 1]] = data[:, 2]
    URM_all = sps.coo_matrix(result)
    return URM_all.tocsr()


def open_file(name):
    try:
        return open("./" + name, 'r')
    except:
        try:
            return open("../datasets/" + name, 'r')
        except:
            try:
                return open("./datasets/" + name, 'r')
            except:
                return open("../../datasets/" + name, 'r')
