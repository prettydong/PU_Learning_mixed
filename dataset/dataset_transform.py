import random

import pandas as pd


def pd_read_csv(path):
    """
    :param path:
    csv path
    :return:
    attributes , grand truth labels
    """
    dataset = pd.read_csv(path)
    grand_truth_labels = dataset['target']
    attributes = dataset.drop('target', 1)
    return attributes, grand_truth_labels


def SCAR_select_fn(gtl,my_random):
    pi = 0.5
    ret = 0
    a = my_random.random()
    # print(a)
    if gtl == 1:
        if a > pi:
            ret = gtl
    return ret


def from_pn2pu(attributes, grand_truth_labels, select_function,rs):
    """
    :param rs:
    Random seed
    :param attributes:
    pandas and not categorical
    :param grand_truth_labels:
    pandas 0/1
    :param select_function:
    should be set into
    :return:
    PU_labels
    """
    PU_labels = []
    my_random = random.Random()
    my_random.seed(rs)
    for i in range(len(attributes)):
        PU_labels.append(select_function(grand_truth_labels[i],my_random))
    PU_labels = pd.Series(PU_labels)
    return PU_labels


def get_PU_dataset(dataset_name,random_seed):
    if dataset_name == 'heart':
        a, gtl = pd_read_csv("../dataset/heart/heart.csv")
        pul = from_pn2pu(a, gtl, SCAR_select_fn,random_seed)
        return a, gtl, pul


if __name__ == '__main__':
    get_PU_dataset('heart')