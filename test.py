from utils import load_data, calculate_acc, memory_print
import numpy as np
from load_CIFAR10 import Load_CIFAR10
from params import Params
import os
import time

print("CALCULATING TEST ACCURACY")
params = Params()
CIFAR = Load_CIFAR10(params)
test_images, test_labels = CIFAR.get_test_set()
print(test_images.shape, test_labels.shape)

ph = load_data(os.path.join(params.save_data, "ph"))

time_start = time.time()
ph_out = ph.transform(test_images)
del test_images

for ratio in params.ratios:
    print("ratio:", ratio)
    concate = []
    for layer in range(0, 2):   ## ATTENTION!!!
        K_transform = load_data(os.path.join(params.save_data, "K_transform_layer{}".format(layer)))
        batch_size = 5000
        con = []
        for b in range(0, ph_out[0].shape[0], batch_size):
            data = ph_out[layer][b: b + batch_size]
            data = K_transform.predict(data)
            # mean1 = np.mean(data, axis=1, keepdims=True)
            # data = mean1 - data
            # data = np.where(data < 0, 0, data)
            # data = np.sum(data, axis=1)
            con.append(data)
        con = np.concatenate(con, axis=0)
        mo3 = load_data(os.path.join(params.save_data, 'simple_pred_layer{}_ratio{}'.format(layer, ratio)))
        prediction = mo3.predict(con)
        calculate_acc(prediction, test_labels)
        concate.append(prediction)


    for layer in range(2, params.num_layers):   ## ATTENTION!!!
        data = ph_out[layer]
        data = np.reshape(data, newshape=(data.shape[0], -1))
        lag = load_data(os.path.join(params.save_data, 'LAG_{}_{}'.format(layer, ratio)))
        lag_pred = lag.predict_proba(data)
        concate.append(lag_pred)
    concate = np.concatenate(concate, axis=1)
    print("Concate shape:", concate.shape)
    rf = load_data(os.path.join(params.save_data, 'RF_{}'.format(ratio)))
    prediction = rf.predict(concate)
    # calculate_acc(prediction, subset_label)
    print("ACC=", np.sum(prediction.reshape((-1)) == test_labels.reshape((-1))) / test_labels.shape[0] * 100)
    print("Time cost - test:", time.time() - time_start)




