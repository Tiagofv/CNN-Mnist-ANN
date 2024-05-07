import numpy as np
import struct
from array import array


# Read MNIST dataset from kaggle
# the files are already inside the mnist folder
# unfortunately lecunn website seems to be requiring auth to download MNIST, so i used the kaggle dataset
class MNISTReader(object):
    def __init__(self, train_img, train_lbl, test_img, test_lbl):
        self.train_img = train_img
        self.train_lbl = train_lbl
        self.test_img = test_img
        self.test_lbl = test_lbl

    def read_data(self, img_path, lbl_path):
        lbls = []
        with open(lbl_path, 'rb') as f:
            m, s = struct.unpack(">II", f.read(8))
            if m != 2049:
                raise ValueError('Invalid magic number: expected 2049, got {}'.format(m))
            lbls = array("B", f.read())

        with open(img_path, 'rb') as f:
            m, s, r, c = struct.unpack(">IIII", f.read(16))
            if m != 2051:
                raise ValueError('Invalid magic number: expected 2051, got {}'.format(m))
            img_data = array("B", f.read())

        imgs = []
        for i in range(s):
            imgs.append([0] * r * c)
        for i in range(s):
            img = np.array(img_data[i * r * c:(i + 1) * r * c])
            img = img.reshape(28, 28)
            imgs[i][:] = img

        return imgs, lbls

    def load(self):
        train_images, train_labels = self.read_data(self.train_img, self.train_lbl)
        test_images, test_labels = self.read_data(self.test_img, self.test_lbl)
        return (train_images, train_labels), (test_images, test_labels)


