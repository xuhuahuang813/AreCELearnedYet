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

input_dim = 11
hidden_dim = 64
output_dim = 10000

class Args:
    def __init__(self, **kwargs):
        self.bs = 32
        self.epochs = 500
        self.lr = 0.001 # default value in both pytorch and keras
        self.hid_units = '64'
        self.bins = 200
        self.train_num = 1000 # 默认验证和测试是训练的1/10

        # overwrite parameters from user
        self.__dict__.update(kwargs)

class TextDataset(Dataset):
    def __init__(self, texts, labels, truecards):
        self.texts = texts
        self.labels = labels
        self.truecards = truecards

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.truecards[idx]

class TextSentimentModel(nn.Module):
    def __init__(self):
        super(TextSentimentModel, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # L.info(text.shape)
        output, _ = self.rnn(text)
        last_hidden_state = output[:, -1, :]
        sentiment_logits = self.fc(last_hidden_state)
        return sentiment_logits
    def name(self):
        return f"lstm_{hidden_dim}"
    
def make_dataset(dataset, num=-1):
    X, y, gt = dataset
    # 将list转为tenser
    X = torch.tensor(X).view(num, 50, 11)
    y = torch.tensor(y)
    gt = torch.tensor(gt).view(num, 50)
    L.info(f"{X.shape}, {y.shape}, {gt.shape}")
    if num <= 0:
        return TextDataset(X, y, gt)
    else:
        return TextDataset(X[:num], y[:num], gt[:num])
    
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
    args = Args(**params)
    
    # 加载数据集，将csv文件转为Table类
    table = load_table(dataset, version)
    
    dataset = load_lstm_dataset(table, workload, seed, params['bins'])
    
    # 加载训练集和验证集
    train_dataset = make_dataset(dataset['train'], int(args.train_num))
    valid_dataset = make_dataset(dataset['valid'], int(args.train_num/10))
    L.info(f"Number of training samples: {len(train_dataset)}")
    L.info(f"Number of validation samples: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.bs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs)
    
    model = TextSentimentModel().to(DEVICE)
    
    state = {
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': table.dataset,
        'version': table.version,
        'workload': workload,
        # 'model_size': model_size,
        # 'fea_num': 11,
    }
    model_path = MODEL_ROOT / table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{table.version}_{workload}-{model.name()}_bin{args.bins}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}.pt"
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_loss = float('inf')

    start_stmp = time.time()
    valid_time = 0
    for epoch in range(args.epochs):
        train_losses = []
        model.train()
        for _, data in enumerate(train_loader):
            inputs, labels, truecards = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())  
        avg_train_loss = sum(train_losses) / len(train_losses)  # 计算平均损失
        dur_min = (time.time() - start_stmp) / 60
        L.info(f"Epoch {epoch+1}, loss: {avg_train_loss}, time since start: {dur_min:.1f} mins")
        
        L.info(f"Test on valid set...")
        valid_stmp = time.time()
        model.eval()
        val_losses = []
        for _, data in enumerate(valid_loader):
            inputs, labels, truecards = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            
            with torch.no_grad():
                preds = model(inputs)
                val_loss = criterion(preds, labels)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        L.info(f"Validation Loss: {avg_val_loss}")
        # TODO 计算metrics，即qerror
        metrics = 1
        
        if avg_val_loss < best_valid_loss:
            L.info('***** best valid loss for now!')
            best_valid_loss = avg_val_loss
            state['model_state_dict'] = model.state_dict()
            state['optimizer_state_dict'] = optimizer.state_dict()
            state['valid_error'] = {workload: metrics}
            state['train_time'] = (valid_stmp-start_stmp-valid_time) / 60
            state['current_epoch'] = epoch
            torch.save(state, model_file)

        valid_time += time.time() - valid_stmp

    L.info(f"Training finished! Time spent since start: {(time.time()-start_stmp)/60:.2f} mins")
    L.info(f"Model saved to {model_file}, best valid: {state['valid_error']}")

        