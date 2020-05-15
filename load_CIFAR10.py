import pickle
import numpy as np

class Load_CIFAR10():
    def __init__(self, params):
        # separate the dataset into different labels to ensure the balance after sampling
        self.params = params

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo, encoding='bytes')
        return dictionary

    def get_train_set(self, ratio):
        separated_train_set = [np.zeros([5000, 32, 32, 3], dtype=np.float) for _ in range(10)]
        current_row = [0 for _ in range(10)]
        for batch_name in self.params.train_batches_filename:
            batch_data = self.unpickle(batch_name)
            batch_images = batch_data[b'data']
            batch_labels = batch_data[b'labels']
            for ind in range(len(batch_labels)):
                image = np.reshape(batch_images[ind], newshape=[3, 32, 32])
                image = np.moveaxis(image, source=0, destination=2)
                label = batch_labels[ind]
                separated_train_set[label][current_row[label]] = image
                current_row[label] += 1
        if self.params.shuffle_train_set == True:
            for train_set in separated_train_set:
                np.random.shuffle(train_set)
        select_num = int(5000 * ratio)
        selected_separated_train_set = [train_set[0:select_num, :] for train_set in separated_train_set]
        train_images = np.concatenate(selected_separated_train_set, axis=0)
        train_labels = []
        for class_num in range(10):
            for i in range(select_num):
                train_labels.append(class_num)
        train_labels = np.array(train_labels).reshape((-1, 1))
        if self.params.shuffle_train_set == True:
            shuffle_state = np.random.get_state()
            np.random.shuffle(train_images)
            np.random.set_state(shuffle_state)
            np.random.shuffle(train_labels)
        return train_images, train_labels

    def get_test_set(self):
        test_data = self.unpickle(self.params.test_batches_filename)
        test_images = np.reshape(test_data[b'data'], newshape=[-1, 3, 32, 32])
        test_images = np.moveaxis(test_images, source=1, destination=3).copy()
        test_labels = np.array(test_data[b'labels']).reshape((-1, 1))
        if self.params.shuffle_test_set == True:
            shuffle_state = np.random.get_state()
            np.random.shuffle(test_images)
            np.random.set_state(shuffle_state)
            np.random.shuffle(test_labels)
        return test_images, test_labels
