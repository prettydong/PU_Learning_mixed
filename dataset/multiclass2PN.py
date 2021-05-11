import pickle

f = open('sk_digits/digits.pkl', 'rb')
data, labels = pickle.load(f)

f.close()
labels = [str(_) for _ in labels]
label_sets = set(labels)
print("label set:",label_sets)

i = input("P label set(use ',' to split):")
p_label_set = i.split(',')
assert (type(p_label_set)==list)
labels = [1 if _ in p_label_set else 0 for _ in labels]

f = open('sk_digits/digitsPN.pkl','wb')
pickle.dump([data,labels],f)
f.close()
