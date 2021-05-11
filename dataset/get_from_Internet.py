from sklearn import datasets
import pickle


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    f = open("sk_digits/digits.pkl", 'wb')
    pickle.dump([data, label], f)
    f.close()
    return True


if __name__ == '__main__':
    get_data()
