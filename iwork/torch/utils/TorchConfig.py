#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'TorchConfig'
__author__ = 'JieYuan'
__mtime__ = '2019/4/17'
"""

import os
import torch
import random
import numpy as np


class TorchConfig(object):
    """Hyper-parameters"""

    input_size = 784
    hidden_size = 500
    num_classes = 10

    num_epochs = 5
    batch_size = 128
    lr = 0.001

    def __init__(self, seed=2019):
        self.seed = seed
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')  # Device configuration
        self.set_seed()
        # torch.set_default_tensor_type(torch.DoubleTensor)

    def set_seed(self):
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_cuda:
            print('GPU: %s' % torch.cuda.get_device_name(0))
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
