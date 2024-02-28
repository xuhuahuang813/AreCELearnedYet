'''
hxh
lstm模型训练及验证
'''
import time
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from ...dataset.dataset import load_table
from .common import load_lstm_dataset
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS

L = logging.getLogger(__name__)

# input_dim = 11
# hidden_dim = 64
# output_dim = 10000

class Args:
    def __init__(self, **kwargs):
        self.bs = 32
        self.epochs = 500
        self.lr = 0.001 # default value in both pytorch and keras
        self.hid_units = '256_256_512_1024_4096'
        self.bins = 200
        self.train_num = 1000 # 默认验证和测试是训练的1/10

        # overwrite parameters from user
        self.__dict__.update(kwargs)

class TextDataset(Dataset):
    def __init__(self, texts, labels, truecards, colList):
        self.texts = texts
        self.labels = labels
        self.truecards = truecards
        self.colList = colList

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.truecards[idx], self.colList[idx]

class TextSentimentModel(nn.Module):
    def __init__(self, hid_units, input_dim=11, output_dim=10000):
        super(TextSentimentModel, self).__init__()
        self.hid_units_list = [int(u) for u in hid_units.split('_')] # hid_units = '64_128_256' 其中64是lstm的隐藏层，其他是线性层。
        
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=self.hid_units_list[0], batch_first=True)
        
        # 中间线性层
        linear_layers = []
        in_hid_size = self.hid_units_list[0]
        for out_hid_size in self.hid_units_list[1: ]:
            linear_layers.append(nn.Linear(in_hid_size, out_hid_size))
            linear_layers.append(nn.ReLU(inplace=True))
            in_hid_size = out_hid_size
        self.linear_layers = nn.Sequential(*linear_layers)
        
        # 最终线性层
        self.final_linear = nn.Linear(self.hid_units_list[-1], output_dim)
        
    def forward(self, text):
        # L.info(text.shape)
        output, _ = self.rnn(text)
        last_hidden_state = output[:, -1, :]
        
        x = self.linear_layers(last_hidden_state)
        sentiment_logits = self.final_linear(x)
        return sentiment_logits
    def name(self):
        hid_unit_str = "_".join(map(str, self.hid_units_list))
        return f"lstm_{hid_unit_str}"
    
def make_dataset(dataset, num=-1):
    X, y, gt, colList = dataset
    # 将list转为tenser
    X = torch.tensor(X).view(num, 50, 11)
    y = torch.tensor(y)
    gt = torch.tensor(gt).view(num, 50)
    L.info(f"{X.shape}, {y.shape}, {gt.shape}")
    if num <= 0:
        return TextDataset(X, y, gt, colList)
    else:
        return TextDataset(X[:num], y[:num], gt[:num], colList[:num])

# TODO 解码模型输出
def decodePreds(inputs, preds, truecards, collist):
    L.info("here")
    return None
    

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
    
    # 加载训练、验证数据集，加载cols_alldomain用于解码
    dataset, cols_alldomain = load_lstm_dataset(table, workload, seed, params['bins'])
    
    # 加载训练集和验证集
    train_dataset = make_dataset(dataset['train'], int(args.train_num))
    valid_dataset = make_dataset(dataset['valid'], int(args.train_num/10))
    L.info(f"Number of training samples: {len(train_dataset)}")
    L.info(f"Number of validation samples: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.bs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs)
    
    model = TextSentimentModel(args.hid_units).to(DEVICE)
    
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
    
    # BCEWithLogitsLoss损失函数，不能使preds趋近于[0, 1]区间中，训练过程趋向[-5, 9]
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    best_valid_loss = float('inf')

    # 记录全部训练过程中的损失
    train_loss_list = []
    valid_loss_list = []
    loss_file = model_path / f"{table.version}_{workload}-{model.name()}_bin{args.bins}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}.png"
    
    start_stmp = time.time()
    valid_time = 0
    for epoch in range(args.epochs):
        train_losses = []
        model.train()
        for _, data in enumerate(train_loader):
            inputs, labels, truecards, collist = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())  
        avg_train_loss = sum(train_losses) / len(train_losses) 
        train_loss_list.append(avg_train_loss)
        dur_min = (time.time() - start_stmp) / 60
        L.info(f"Epoch {epoch+1}, loss: {avg_train_loss}, time since start: {dur_min:.1f} mins, lr: {optimizer.param_groups[0]['lr']}")
        # 在每个epoch结束时更新学习率
        # scheduler.step()

        L.info(f"Test on valid set...")
        valid_stmp = time.time()
        model.eval()
        val_losses = []
        for _, data in enumerate(valid_loader):
            inputs, labels, truecards, collist = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            
            with torch.no_grad():
                preds = model(inputs)
                val_loss = criterion(preds, labels)
                val_losses.append(val_loss.item())
                # 计算每个inputs在preds上的cardinality estimation与truecards的差（q-error）
                decodePreds(inputs, preds, truecards, collist)

        avg_val_loss = sum(val_losses) / len(val_losses)
        valid_loss_list.append(avg_val_loss)
        L.info(f"Validation Loss: {avg_val_loss}")
        # 动态调整学习率
        scheduler.step(avg_val_loss)
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
    L.info(f"Model saved to {model_file}, best valid: {state['valid_error']}, best valid loss: {best_valid_loss}")

    # train和valid使用同一个y轴
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(valid_loss_list, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(loss_file)
    
    '''
    # train和valid使用两个y轴
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss_list, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(valid_loss_list, label='Validation Loss', color='red')
    ax2.set_ylabel('Validation Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(loss_file)
    '''
        