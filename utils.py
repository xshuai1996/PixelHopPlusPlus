import psutil
import pickle
import numpy as np

def memory_print():
    print("MEMORY USAGE:", psutil.virtual_memory().used / 1024 / 1024 / 1024, "GB.")

def save_data(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)

def load_data(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    return data

def calculate_acc(prob_predict, labels):
    prob_predict = np.argmax(prob_predict, axis=1)
    corrects = prob_predict.reshape((-1)) == labels.reshape((-1))
    corrects = np.sum(corrects)
    print("Accuracy: ", corrects / labels.reshape(-1).shape[0] * 100, "%")