from __future__ import absolute_import
from __future__ import print_function

import random
import os
import torch

import numpy as np

from torchmimic.data.preprocessing import Discretizer, Normalizer
from torchmimic.data.readers import InHospitalMortalityReader
from torchmimic.data.utils import read_chunk
from torchmimic.data.base_dataset import BaseDataset


class IHMDataset(BaseDataset):
    """
    In-Hospital-Mortality dataset that can be directly used by PyTorch dataloaders. This class preprocessing the data the same way as "Multitask learning and benchmarking with clinical time series data": https://github.com/YerevaNN/mimic3-benchmarks

    :param root: directory where data is located
    :type root: str
    :param train: if true, the training split of the data will be used. Otherwise, the validation dataset will be used
    :type train: bool
    :param n_samples: number of samples to use. If None, all the data is used
    :type steps: int
    :param customListFile: listfile to use. If None, use train_listfile.csv
    :type steps: str
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        n_samples=None,
        customListFile=None,
    ):
        """
        Initialize IHMDataset

        :param root: directory where data is located
        :type root: str
        :param train: if true, the training split of the data will be used. Otherwise, the validation dataset will be used
        :type train: bool
        :param n_samples: number of samples to use. If None, all the data is used
        :type steps: int
        :param customListFile: listfile to use. If None, use train_listfile.csv
        :type steps: str
        """
        super().__init__(transform=transform)

        listfile = "train_listfile.csv" if train else "val_listfile.csv"

        if customListFile is not None:
            listfile = customListFile
            
        print(listfile)

        self._read_data(root, listfile)
        self._load_data(n_samples)

        self.n_samples = len(self.data)

    def _read_data(self, root, listfile):
        self.reader = InHospitalMortalityReader(
            dataset_dir=os.path.join(root, "train"),
            listfile=os.path.join(root, listfile),
        )

        
