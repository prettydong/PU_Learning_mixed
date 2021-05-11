"""
function should be attributes pulabels and other hyper parameters
the fnc is based on PUL knn
"""

import numpy as np
import pandas as pd

from dataset.dataset_transform import get_PU_dataset
from sklearn.metrics import f1_score, accuracy_score


def knn_PU(attributes, pu_labels, k, T):
    """
    :param attributes:
    :param pu_labels:
    :param k:
    :param T:
    :return:
    the fn only use in a small dataset,
    because the calculate of knn is implemented by Python
    all vector should be type of Pandas
    """
    # step 1 -> choose reliable Negative samples

    # all1 = pd.concat([a,pul],1)
    # all1 = all1.rename(columns={0:'pu_labels'})

    attributes = (attributes - attributes.mean()) / attributes.std()
    # print(attributes)
    p_idx_list = pu_labels[pu_labels == 1].index
    u_idx_list = pu_labels[pu_labels != 1].index

    pseudo_p_idx_list = []
    q = 0
    for u_idx in u_idx_list:
        # print(u_idx)
        ui = attributes.iloc[u_idx]
        sim_list = []
        for p_idx in p_idx_list:
            pi = attributes.iloc[p_idx]
            # print(ui,pi)
            cos_similar = np.dot(ui, pi) / ((np.linalg.norm(ui) * (np.linalg.norm(pi)))+0.0001)
            sim_list.append(cos_similar)
        sim_list = sorted(sim_list, reverse=True)
        sum_sim = 0
        for i in range(k):
            sum_sim = sum_sim + sim_list[i]

        if sum_sim > T:
            pseudo_p_idx_list.append(u_idx)
            u_idx_list.drop(u_idx)
            q += 1
    print(q)
    # print(pseudo_p_idx_list)
    return pseudo_p_idx_list


if __name__ == '__main__':
    a, gtl, pul = get_PU_dataset('digits', random_seed=1)
    pseudo_p = knn_PU(a, pul, 5, 3.5)
    pseudo_p = pd.Series([1 if i in pseudo_p else 0 for i in range(len(gtl))])
    pred = pul + pseudo_p
    print(accuracy_score(pred, gtl))
