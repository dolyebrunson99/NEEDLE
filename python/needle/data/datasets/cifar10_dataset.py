import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        self.train = train
        if train:
            self.img = np.empty(shape=(50000, 3072),
                                dtype=np.uint8)
            self.lbl = np.empty(shape=(50000,),
                                dtype=np.float32)
            # Training data are packed into 5 files.
            # see https://www.cs.toronto.edu/~kriz/cifar.html
            for i in range(5):
                with open(os.path.join(base_folder, f'data_batch_{i+1}'), 'rb') as f:
                    training_batch = pickle.load(f, encoding='bytes')
                self.img[i * 10000 : (i+1) * 10000] = training_batch[b'data']
                self.lbl[i * 10000 : (i+1) * 10000] = np.array(training_batch[b'labels'],
                                                               dtype=np.float32)
        else:
            with open(os.path.join(base_folder, 'test_batch'), 'rb') as f:
                test_set = pickle.load(f, encoding='bytes')
            self.img = test_set[b'data']
            self.lbl = np.array(test_set[b'labels'],
                                dtype=np.float32)
        self.transforms = transforms
        self.transforms_fn = lambda I, singleton : \
                                    self.apply_transforms(I.astype(np.float32).reshape((3, 32, 32) if singleton \
                                                                                                   else (-1, 3, 32, 32)) / 255.)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        return self.transforms_fn(self.img[index],
                                  isinstance(index, int)), self.lbl[index]

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return 50000 if self.train else 10000
