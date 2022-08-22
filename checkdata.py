from json import load
import pickle 
import matplotlib.pyplot as plt


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


data = load_pickle_file('cache/coded_sps_norm.pickle')

print(data[0].shape)

# plt.imshow(data[0])
# plt.title(data[0].shape)
# plt.show()