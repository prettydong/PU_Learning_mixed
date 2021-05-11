# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from dataset.dataset_transform import get_PU_dataset
from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target

    return data, label


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color='r' if label[i] == 0 else 'b',
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def show_data_tSNE(data, label, normalize=False):
    # data, label, w = get_PU_dataset('heart', random_seed=1)
    if normalize:
        data = (data - data.mean()) / data.std()
    print(data, label)
    print('Computing t-SNE embedding')
    t_sne = TSNE(n_components=2, init='pca', random_state=2, perplexity=30, method='exact')
    t0 = time()
    result = t_sne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show()


if __name__ == '__main__':
    data, label, w = get_PU_dataset('heart', random_seed=1)
    show_data_tSNE(data,label)
