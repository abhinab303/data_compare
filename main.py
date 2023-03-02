import pdb

import numpy as np
import os

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader

import tensorflow_datasets as tfds
import time


def one_hot(labels, num_classes, dtype=np.float32):
    return (labels[:, None] == np.arange(num_classes)).astype(dtype)


def sort_by_class(X, Y):
    sort_idxs = Y.argmax(1).argsort()
    X, Y = X[sort_idxs], Y[sort_idxs]
    return X, Y

def normalize_cifar10_images(X):
    mean_rgb = np.array([[[[0.4914 * 255, 0.4822 * 255, 0.4465 * 255]]]], dtype=np.float32)
    std_rgb = np.array([[[[0.2470 * 255, 0.2435 * 255, 0.2616 * 255]]]], dtype=np.float32)
    X = (X.astype(np.float32) - mean_rgb) / std_rgb
    return X

dataset_dir = "/Users/abhinabacharya/PycharmProjects/data_compare"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class IndexedDataset(Dataset):
    def __init__(self):
        self.cifar10 = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    def __getitem__(self, index):
        data, target = self.cifar10[index]
        # Your transformations here (or set it in CIFAR10)
        return data, target, index

    def __len__(self):
        return len(self.cifar10)


indexed_dataset = IndexedDataset()
indexed_loader = DataLoader(
    indexed_dataset,
    batch_size=1, shuffle=True,
    num_workers=1, pin_memory=True)

def load_cifar10_data_diet():
    # load cifar10
    print('load cifar10... ', end='')
    time_start = time.time()
    (X_train, Y_train), (X_test, Y_test) = tfds.as_numpy(tfds.load(
      name='cifar10', split=['train', 'test'], data_dir=args.data_dir,
      batch_size=-1, download=False, as_supervised=True))
    print(f'{int(time.time() - time_start)}s')
    # normalize images, one hot labels
    num_classes = 10
    X_train, X_test = normalize_cifar10_images(X_train), normalize_cifar10_images(X_test)
    Y_train, Y_test = one_hot(Y_train, num_classes), one_hot(Y_test, num_classes)
    # sort by class
    X_train, Y_train = sort_by_class(X_train, Y_train)
    X_test, Y_test = sort_by_class(X_test, Y_test)

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_cifar10_data_diet()
data_diet_data = [ x for x in indexed_loader]

pdb.set_trace()