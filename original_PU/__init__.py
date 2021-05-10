import numpy
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from dataset.dataset_transform import get_PU_dataset
from sklearn.metrics import f1_score, accuracy_score


def GNB_PN(attributes, ground_truth_labels):
    gnb = GaussianNB()
    gnb.fit(attributes, ground_truth_labels)
    return gnb.predict(attributes)


def GNB_PU(attributes, pu_labels):
    gnb = GaussianNB()
    gnb.fit(attributes, pu_labels)
    return gnb.predict(attributes)


def two_step_GNB(attributes, ground_truth_labels, pu_labels, max_iteration):
    """
    :param attributes:
    should be a list of attributes vector
    :param ground_truth_labels:
    should be a int type 1> P,0 > U
    :param max_iteration:
    set the max of iteration
    :param pu_labels:
    should be a int type 1> P,0 > U
    :return labels:
    the pred labels will be return\
    the GNB with PN : 85%
    the GNB with PU 1step : 75%
    """

    gnb = GaussianNB()
    gnb.fit(attributes, pu_labels)
    pred = gnb.predict(attributes)

    for i in range(max_iteration):
        pred_increase = pu_labels + pred
        pred_increase = pd.Series([1 if i > 0 else 0 for i in pred_increase])
        gnb.fit(attributes, pred_increase)
        pred = gnb.predict(attributes)

    return pred


if __name__ == '__main__':
    a, gtl, pul = get_PU_dataset('heart',random_seed = 1)
    result_GNB_PU = numpy.array([accuracy_score(GNB_PU(a, pul), gtl) for i in range(100)])
    result_GNB_PN = numpy.array([accuracy_score(GNB_PU(a, gtl), gtl) for i in range(100)])
    result_GNB_PU2S = numpy.array([accuracy_score(two_step_GNB(a, gtl, pul, 1000), gtl) for i in range(100)])
    print("result_GNB_PN:" + str(result_GNB_PN.mean()))
    print("result_GNB_PU:" + str(result_GNB_PU.mean()))
    print("result_GNB_PU2S:" + str(result_GNB_PU2S.mean()))
