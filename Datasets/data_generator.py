#! /bin/python
# -*- coding: utf-8 -*-
#
#  data_generator.py
#
#  this script is to generate a random dataset from the originial one

import numpy as np
import pandas as pd


def data_generator(data):
    """

    @param data:
    @param seed:
    @return:
    """
    y = data['outcome']
    X = data.drop(['outcome'],axis=1)
    indexes_cols_drop = np.random.randint(len(X.columns),size=10)
    columns = list(X.columns)
    X = X.drop([columns[i] for i in range(len(columns)) if i in indexes_cols_drop],axis=1)
    nb_rows = np.random.randint(7000,len(X))
    indexes_rows = np.random.randint(len(X),size = nb_rows)
    data_for_project = pd.concat([X,y],axis=1)
    data_for_project = data_for_project.loc[indexes_rows,:]
    return data_for_project


if __name__ == '__main__':
    np.random.seed(21)
    data = pd.read_csv("Dataset_project_RS.csv", index_col=0)
    data_for_project = data_generator(data)
