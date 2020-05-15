import os


class Params(object):
    def __init__(self):
        """ Wrapper class for various parameters. """
        dataset_path = os.path.join("cifar-10-python", "cifar-10-batches-py")
        self.train_batches_filename = [os.path.join(dataset_path, "data_batch_1"),
                                       os.path.join(dataset_path, "data_batch_2"),
                                       os.path.join(dataset_path, "data_batch_3"),
                                       os.path.join(dataset_path, "data_batch_4"),
                                       os.path.join(dataset_path, "data_batch_5")]
        self.test_batches_filename = os.path.join(dataset_path, "test_batch")
        self.shuffle_train_set = True
        self.shuffle_test_set = False
        self.data_use_ratio_PixelHop = 0.2
        self.data_use_ratio_feature_selection = 1

        self.save_data = os.path.join("save_data")
        self.SaabArgs = [{'num_AC_kernels': -1, 'needBias': False, 'useDC': True, 'cw': False},
                         {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'cw': True},
                         {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'cw': True},
                         {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'cw': True}]
        self.num_layers = len(self.SaabArgs)
        self.neighborArgs = [{'kernel': (1, 3, 3, 1), 'stride': (1, 1, 1, 1)},
                             {'kernel': (1, 3, 3, 1), 'stride': (1, 1, 1, 1)},
                             {'kernel': (1, 3, 3, 1), 'stride': (1, 1, 1, 1)},
                             {'kernel': (1, 3, 3, 1), 'stride': (1, 1, 1, 1)}]
        self.poolingArg = [{'win': 2, 'pad': False},
                           {'win': 2, 'pad': True},
                           {'win': 2, 'pad': True},
                           {'win': 1, 'pad': False}]
        self.ratios = [1/32, 1/16, 1/8, 1/4, 1]
        self.Ns = 6000
