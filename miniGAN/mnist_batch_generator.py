import os
import subprocess
import pickle
import gzip
import numpy as np
from random import getrandbits


class MNIST_loader():

    def __init__(self, mnist_path, batch_size):

        if not os.path.isfile(os.path.join(mnist_path, 'mnist.pkl.gz')):

            print('downloading mnist..')

            download_cmd = 'wget -N http://deeplearning.net/data/mnist/mnist.pkl.gz -P {}'.format(mnist_path)

            print(download_cmd)

            subprocess.call(download_cmd, shell=True)

        self.mnist_path = mnist_path
        self.batch_size = batch_size

        self.image_size = 28
        self.image_channels = 1

    @staticmethod
    def batch_gen(X, y, N):

        shape = (N, 1, 28, 28)
        while True:
            idx = np.random.choice(len(y), N)
            images = X[idx].astype('float32')
            gt = y[idx].astype('int32')
            yield {'data': np.reshape(images, shape), 'seg': np.reshape(images.copy(), shape), 'class':gt}

    @staticmethod
    def salt_and_pepper(batch_gen, max_perc):

        rng = np.random.RandomState(seed=42)

        for item in batch_gen:

            shape = item['data'].shape

            batch_size, ch, rows, cols = shape[0], shape[1], shape[2], shape[3]
            # number of pixels
            mp = rows * cols * ch

            for ix in range(batch_size):

                num = np.round(mp * rng.uniform(low=0.001, high=max_perc)).astype(np.int32)
                rand_ixs = rng.randint(low=0, high=mp, size=num)
                noise_ixs =  [getrandbits(1) for x in range(num)]

                data_f = item['data'][ix].flatten()
                data_f[rand_ixs] = noise_ixs
                item['data'][ix] = data_f.reshape((ch, rows, cols))

            yield item

    @staticmethod
    def get_BHWC(batch_gen):

        for item in batch_gen:

            item['data'] = np.transpose(item['data'], (0,2,3,1))
            item['seg'] = np.transpose(item['seg'], (0,2,3,1))

            yield item


    def get_batch_loader(self, max_perc):

        train, val, test = pickle.load(gzip.open(os.path.join(self.mnist_path, 'mnist.pkl.gz')), encoding='latin1', )

        X_train, y_train = train
        X_valid, y_valid = val

        self.no_train_batches = len(X_train) // self.batch_size
        self.no_valid_batches = len(X_valid) // self.batch_size

        train_gen = self.batch_gen(X_train, y_train, self.batch_size)
        train_gen = self.salt_and_pepper(train_gen, max_perc)
        train_gen = self.get_BHWC(train_gen)

        valid_gen = self.batch_gen(X_valid, y_valid, self.batch_size)
        valid_gen = self.salt_and_pepper(valid_gen, max_perc)
        valid_gen = self.get_BHWC(valid_gen)

        return train_gen, valid_gen






