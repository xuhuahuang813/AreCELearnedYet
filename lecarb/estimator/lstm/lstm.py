'''
hxh
lstm模型训练及验证
'''
import time
import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ...dataset.dataset import load_table
from .common import load_lstm_dataset
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS

L = logging.getLogger(__name__)

def train_lstm(seed, dataset, version, workload, params, sizelimit):
    L.info(f"training LSTM model with seed {seed}")
    
    # uniform thread number
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置lstm模型参数
    L.info(f"params: {params}")
    # args = Args(**params)
    
    # 加载数据集，将csv文件转为Table类
    table = load_table(dataset, version)
    
    dataset = load_lstm_dataset(table, workload, seed, params['bins'])
    
    # 加载训练集
    print("here")
    
    