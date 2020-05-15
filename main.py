from params import Params
from load_CIFAR10 import Load_CIFAR10
from pixelhop2 import Pixelhop2
import time
import numpy as np
from utils import save_data, load_data, calculate_acc
import os
from llsr import LLSR as myLLSR
from lag import LAG
from module3 import RF
from cluster_count import K_means_transform
from new_3 import Decision

if __name__ == "__main__":

    def load_params():
        time_start = time.time()
        params = Params()
        print("Time cost - load params:", time.time() - time_start)
        return params

    def load_CIFAR_train(params):
        time_start = time.time()
        CIFAR10 = Load_CIFAR10(params)
        train_images, train_labels = CIFAR10.get_train_set(ratio=params.data_use_ratio_PixelHop)
        print("Time cost - load CIFAR10:", time.time() - time_start)
        print(train_images.shape, train_labels.shape)
        return train_images

    def module1(params, train_images):
        time_start = time.time()
        ph = Pixelhop2(TH1=0.0001, TH2=0.0001, SaabArgs=params.SaabArgs, neighborArgs=params.neighborArgs, poolingArg=params.poolingArg)
        ph.fit(train_images)
        print("Time cost - PixelHop++ units:", time.time() - time_start)
        del train_images
        # save data to files
        save_data(ph, os.path.join(params.save_data, "ph"))

    def transfer_all_data(params):
        # load data from files
        ph = load_data(os.path.join(params.save_data, "ph"))
        # feed all data to pixel hop units
        CIFAR10 = Load_CIFAR10(params)
        all_train_images, all_train_labels = CIFAR10.get_train_set(ratio=1)
        print(all_train_images.shape)
        batch_size = 5000
        for i in range(0, all_train_images.shape[0], batch_size):
            print(i,"-", i + batch_size)
            out_i = ph.transform(all_train_images[i: i + batch_size])
            for j in range(params.num_layers):
                save_data(out_i[j], os.path.join(params.save_data, 'out_{}_{}'.format(i, j)))

        del all_train_images
        save_data(all_train_labels, os.path.join(params.save_data, "all_train_labels"))
        print("All transfered data saved")

    def k_means_for_first_layers(params):
        for layer in range(0, 2):   ## ATTENTION!!!
            time_start = time.time()
            batch_size = [5000 * 15 * 15, 5000 * 7 * 7]
            num_cluster = [100, 150]
            K_transform = K_means_transform(batch_size=batch_size[layer], num_cluster=num_cluster[layer])
            for j in range(0, 50000, 5000):
                a_batch = load_data(os.path.join(params.save_data, 'out_{}_{}'.format(j, layer)))
                K_transform.batch_fit(a_batch)
            print("Time cost - K_means FIT, layer{}:".format(layer), time.time() - time_start)
            time_start = time.time()
            for j in range(0, 50000, 5000):
                a_batch = load_data(os.path.join(params.save_data, 'out_{}_{}'.format(j, layer)))
                a_batch = K_transform.predict(a_batch)
                print("batch shape after transform:", a_batch.shape)
                save_data(a_batch, os.path.join(params.save_data, "K_transform_batch_layer{}_batch{}".format(layer, j)))
            save_data(K_transform, os.path.join(params.save_data, "K_transform_layer{}".format(layer)))
            print("Time cost - K_means transform, layer{}:".format(layer), time.time() - time_start)

    def simple_pred(params):
        all_train_labels = load_data(os.path.join(params.save_data, "all_train_labels"))
        all_train_labels = all_train_labels.reshape((-1))
        for ratio in params.ratios:
            time_start = time.time()
            subset_label = all_train_labels[:int(all_train_labels.shape[0] * ratio)]
            for layer in range(0, 2):   ## ATTENTION!!!
                concate = []
                for j in range(0, 50000, 5000):
                    a_batch = load_data(os.path.join(params.save_data, "K_transform_batch_layer{}_batch{}".format(layer, j)))
                    concate.append(a_batch)
                concate = np.concatenate(concate, axis=0)
                concate = concate[:int(all_train_labels.shape[0] * ratio)]
                # num_PCA_kernels = [80, 125]
                mo3 = Decision()
                mo3.fit(concate, subset_label)
                prediction = mo3.predict(concate)
                save_data(prediction, os.path.join(params.save_data, "sinple_predict_ratio{}_layer{}".format(ratio, layer)))
                calculate_acc(prediction, subset_label)
                save_data(mo3, os.path.join(params.save_data, 'simple_pred_layer{}_ratio{}'.format(layer, ratio)))
            print("Time cost - simple_pred:", time.time() - time_start)

    def lag_module(params, use_filters):
        all_train_labels = load_data(os.path.join(params.save_data, "all_train_labels"))
        for ratio in params.ratios:
                time_start = time.time()
                if use_filters is True:
                    filters = load_data(os.path.join(params.save_data, "filters_" + str(ratio)))
                subset_label = all_train_labels[:int(all_train_labels.shape[0] * ratio)].copy()
                for i in range(2, params.num_layers):       ## ATTENTION
                    data = []
                    for j in range(0, 50000, 5000):
                        data.append(load_data(os.path.join(params.save_data, 'out_{}_{}'.format(j, i))))
                    data = np.concatenate(data, axis=0)
                    subset_data = data[:int(data.shape[0]*ratio)].copy()
                    del data

                    subset_data = np.reshape(subset_data, newshape=(subset_data.shape[0], -1))
                    print("subset data shape:", subset_data.shape)
                    if use_filters is True:
                        print("Number of selected channels:", np.sum(filters[i]))
                        subset_data = subset_data[:, filters[i]]

                    lag = LAG(num_clusters=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10], alpha=10, learner=myLLSR(onehot=False))
                    lag.fit(subset_data, subset_label)
                    subset_predprob = lag.predict_proba(subset_data)
                    save_data(lag, os.path.join(params.save_data, 'LAG_{}_{}'.format(i, ratio)))
                    save_data(subset_predprob, os.path.join(params.save_data, 'lag_predict_{}_{}'.format(i, ratio)))
                    print("RATIO=",ratio," LAYER=", i, " DONE")
                print("Time cost - LAG:", time.time() - time_start, "ratio=", ratio)
                save_data(subset_label, os.path.join(params.save_data, 'lag_labels_{}_{}'.format(i, ratio)))

    def final(params):
        all_train_labels = load_data(os.path.join(params.save_data, "all_train_labels"))
        all_train_labels = all_train_labels.reshape((-1))
        for ratio in params.ratios:
            concate = []
            for i in range(0, 2):
                simple_pred = load_data(os.path.join(params.save_data, "sinple_predict_ratio{}_layer{}".format(ratio, i)))
                concate.append(simple_pred)
            for i in range(2, params.num_layers):       ## ATTENTION
                subset_predprob = load_data(os.path.join(params.save_data, 'lag_predict_{}_{}'.format(i, ratio)))
                concate.append(subset_predprob)
            concate = np.concatenate(concate, axis=1)
            print("Concate shape:", concate.shape)
            time_start = time.time()
            subset_label = all_train_labels[:int(all_train_labels.shape[0] * ratio)].copy()

            rf = RF()
            rf.fit(concate, subset_label)
            prediction = rf.predict(concate)
            print("ACC=", np.sum(prediction.reshape((-1)) == subset_label.reshape((-1))) / subset_label.shape[0] * 100)
            # calculate_acc(prediction, subset_label)
            save_data(rf, os.path.join(params.save_data, 'RF_{}'.format(ratio)))
            print("Time cost - RF:", time.time() - time_start, "ratio=", ratio)



    params = load_params()
    # train_images = load_CIFAR_train(params)
    # module1(params, train_images)
    # del train_images
    # transfer_all_data(params)
    # k_means_for_first_layers(params)
    # simple_pred(params)
    lag_module(params, use_filters=False)
    final(params)
