'''
hxh
lstm模型训练及验证

TODO 
【√】query编码和联合域global_cols_alldomain不对应
【√】lstm层数量
【√】args.hid_units
【√】loss function
【√】测试函数
'''
import time
import csv
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
# from tqdm import tqdm
# import concurrent.futures
import multiprocessing
from functools import partial
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from ...dataset.dataset import load_table
from .common import load_lstm_dataset
from ...constants import DEVICE, MODEL_ROOT, NUM_THREADS, RESULT_ROOT

L = logging.getLogger(__name__)

global_table = None # census表信息
global_cols_alldomain = None # 联合域信息 ./common.py中的clos_alldomain
global_cols_alldomain = None # 联合域信息 ./common.py中的clos_alldomain

global_epoch = 0


'''
模型超参
'''
class Args:
    def __init__(self, **kwargs):
        self.bs = 32
        self.epochs = 100
        self.lr = 0.001 
        self.hid_units = '256_1024_4096'
        self.bins = 200 # 用于路径命名，没有实际作用
        self.train_num = 1000 # 默认验证和测试是训练的1/10
        self.lossfunc = 'MSELoss'

        # 使用传入参数，重写超参
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


'''
lstm模型
'''
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


'''
打印q_error百分位点
q_error_list: 单个epoch内所有q_error
'''
def Q_error_print(q_error_list):
    # 直接打印百分比
    # sorted_q_error_list = np.sort(q_error_list)
    # percentiles = [25, 50, 75, 90]
    # percentile_values = np.percentile(sorted_q_error_list, percentiles)
    # for p, value in zip(percentiles, percentile_values):
    #     L.info(f"{p}th percentile: {value}")


    # 只打印非inf数值的百分比
    # q_error_list_noinf = [value.item() for value in q_error_list if value.item() != float('inf')]
    # 按理说 q_error_list中不会存在None和inf
    q_error_list_noinf = [value.item() for value in q_error_list if value is not None and value.item() != float('inf')]
    # sorted_q_error_list = np.sort(q_error_list_noinf)
    # percentiles = [25, 50, 75, 90]
    # percentile_values = np.percentile(sorted_q_error_list, percentiles)
    # for p, value in zip(percentiles, percentile_values):
    #     L.info(f"{p}th percentile: {value}")
    L.info(f"Number of 'inf' values: {len(q_error_list) - len(q_error_list_noinf)}, Number of non 'inf' values: {len(q_error_list_noinf)}")
    
    
    q_error_list = q_error_list_noinf
    # 直接打印，对齐lw_nn
    metrics = {
        '\n25th': np.percentile(q_error_list, 25),
        '50th': np.percentile(q_error_list, 50),
        '75th': np.percentile(q_error_list, 75),
        '90th': np.percentile(q_error_list, 90),
        '95th': np.percentile(q_error_list, 95),
        '99th': np.percentile(q_error_list, 99),
        'max': np.max(q_error_list),
        'mean': np.mean(q_error_list),
    }
    formatted_metrics = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
    L.info(formatted_metrics)

'''
解码模型输出
inputs.shape: torch.Size([32, 50, 11])
preds.shape: torch.Size([32, 10000])
collist [50]
coll 'age/capital_loss'
'''
'''
def decodePreds(inputs, preds, truecards, collist):
    # return [torch.tensor(2)]
    global global_table 
    global global_cols_alldomain 
    
    latest_inputs = inputs[:, -1, :].to('cpu') # 获取最新的查询。latest_inputs获取最外层32个[50, 11]维数组中，每个50个11维数组的最后一个11维数组。latest_inputs[32, 11]
    latest_truecards = truecards[:, -1].to('cpu') # 获取最新的truecards
    preds = preds.to('cpu') 
    
    # limit_q_error_num = 3
    q_error_list = []
    
    total_iterations = len(latest_inputs) # 总计算数量，用于tqdm进度展示
    # for q_error_i, (latest_in, latest_tc, coll, pred_cumulate) in tqdm(enumerate(zip(latest_inputs, latest_truecards, collist, preds)), total=total_iterations, desc="Processing"):
    for q_error_i, (latest_in, latest_tc, coll, pred_cumulate) in enumerate(zip(latest_inputs, latest_truecards, collist, preds)):
        # TODO 为了加速计算，每一个iteration只算limit_q_error_num个query的q-error
        # if q_error_i >= limit_q_error_num:
        #     break
        # TODO 为了加速计算所以continue
        if "capital_loss" in coll:
            continue
        
        # L.info(f"{coll}")
        coll_list = coll.split("/") # coll 'age/capital_loss'
        col_df = global_cols_alldomain[coll] # col_df是coll(age/capital_loss')对应的联合域信息
        
        domain_list = [] # 存储所有满足query谓词的联合域的域位置。即10000维数组中满足query的index。
        for _, row in col_df.iterrows():
            row_in_domain = True
            for coll_list_index in range(len(coll_list)):
                if(row[coll_list[coll_list_index]] < latest_in[(coll_list_index + 1) *2 - 1] or row[coll_list[coll_list_index]] > latest_in[(coll_list_index + 1)*2]):
                    row_in_domain = False
                if row_in_domain == False:
                    break
            if row_in_domain == True:
                domain_list.append(row["index_alldomain"])
            
        pred = [pred_cumulate[i] - pred_cumulate[i-1] if i > 0 else pred_cumulate[i] for i in range(len(pred_cumulate))] # pred_cumulate是累计概率。pred是当前分位点的概率。
        domain_list_sum = np.sum([pred[int(i)] for i in domain_list]) # domain_list上所有分位点概率和
        estimate_card = domain_list_sum * global_table.row_num # 预估的基数
        q_error = Q_error(estimate_card, latest_tc) # 计算q_error
        # L.info(type(q_error))
        # L.info(f"estimate {estimate_card}; true {latest_tc}; q-error {q_error}")
        q_error_list.append(q_error)           
    return q_error_list
'''


'''
多进程decodePreds
'''
def process_iteration(args):
    global global_epoch
    
    q_error_i, (latest_in, latest_tc, coll, pred_cumulate) = args
    global global_table 
    global global_cols_alldomain 
    
    if "capital_loss" in coll:
        return None
    
    coll_list = coll.split("/")
    col_df = global_cols_alldomain[coll]
    
    domain_list = []
    for _, row in col_df.iterrows():
        row_in_domain = True
        for coll_list_index in range(len(coll_list)):
            if (row[coll_list[coll_list_index]] < latest_in[(coll_list_index + 1) * 2 - 1] or
                    row[coll_list[coll_list_index]] > latest_in[(coll_list_index + 1) * 2]):
                row_in_domain = False
            if not row_in_domain:
                break
        if row_in_domain:
            domain_list.append(row["index_alldomain"])
            
    pred = [pred_cumulate[i] - pred_cumulate[i-1] if i > 0 else pred_cumulate[i] for i in range(len(pred_cumulate))]
    domain_list_sum = np.sum([pred[int(i)] for i in domain_list])
    estimate_card = domain_list_sum * global_table.row_num
    q_error = Q_error(estimate_card, latest_tc)
    
    if global_epoch >= 198 and torch.isinf(q_error):
        L.info(f"\n【inf】【coll】{coll}")
    elif global_epoch >= 198 and q_error > 3.0:
        L.info(f"\n【2】【coll】{coll}【Q】{q_error}")
    return q_error, estimate_card

'''
多进程decodePreds 创建进程池
'''
def decodePreds_pool(inputs, preds, truecards, collist):
    latest_inputs = inputs[:, -1, :].to('cpu')
    latest_truecards = truecards[:, -1].to('cpu')
    preds = preds.to('cpu') 
    
    q_error_list = []
    estimate_list = []
    total_iterations = len(latest_inputs)
    
    with multiprocessing.Pool() as pool:
        func_partial = partial(process_iteration)
        args = list(enumerate(zip(latest_inputs, latest_truecards, collist, preds)))
        results = pool.map(func_partial, args)
    
    q_error_list, estimate_list = zip(*results)
    return list(q_error_list), list(estimate_list)


'''
计算q_error
estimate_card: 预估的基数
true_card: 真实基数
'''
def Q_error(estimate_card, true_card):
    # 如果预测值是负数，则认为预测值是1
    # if estimate_card <= 0:
    #     estimate_card = 1
    # return max(estimate_card/true_card, true_card/estimate_card)

    # 如果预测的是负数，则qerror直接返回表行数（最大值）
    if estimate_card < 0:
        return torch.tensor(float('inf'))
        # estimate_card = torch.tensor([1.0])
    if estimate_card == 0:
        estimate_card = torch.tensor([1.0])
    return max(estimate_card/true_card, true_card/estimate_card)
        

'''
模型训练
'''
def train_lstm(seed, dataset, version, workload, params, sizelimit):
    global global_table 
    global global_cols_alldomain 
    global global_epoch
    global global_epoch_all
    
    L.info(f"training LSTM model with seed {seed}")
    
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"torch threads: {torch.get_num_threads()}")
    
    # 固定随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置lstm模型参数
    L.info(f"params: {params}")
    args = Args(**params)
    
    # 加载数据集，将csv文件转为Table类
    global_table = load_table(dataset, version)
    
    # 加载训练、验证数据集，加载cols_alldomain用于解码
    Dataset, global_cols_alldomain = load_lstm_dataset(global_table, workload, seed, params['bins'])
    
    # 加载训练集和验证集
    train_dataset = make_dataset(Dataset['train'], int(args.train_num))
    valid_dataset = make_dataset(Dataset['valid'], int(args.train_num/10))
    L.info(f"Number of training samples: {len(train_dataset)}")
    L.info(f"Number of validation samples: {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.bs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.bs)
    
    # 设置模型
    model = TextSentimentModel(args.hid_units).to(DEVICE)
    state = {
        'seed': seed,
        'args': args,
        'device': DEVICE,
        'threads': torch.get_num_threads(),
        'dataset': global_table.dataset,
        'version': global_table.version,
        'workload': workload,
        # 'model_size': model_size,
        # 'fea_num': 11,
    }
    model_path = MODEL_ROOT / global_table.dataset
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / f"{global_table.version}_{workload}-{model.name()}_loss{args.lossfunc}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}.pt"
    
    # 设置损失函数、优化器、自适应学习率
    if(args.lossfunc == 'MSELoss'):
        criterion = nn.MSELoss() # MSELoss比L1Loss好
    elif(args.lossfunc == 'SmoothL1Loss'):
        criterion = nn.SmoothL1Loss() # 和MSELoss差不多。在AllDomain上，SmoothL1Loss比MSELoss差很多
    elif(args.lossfunc == 'L1Loss'):
        criterion = nn.L1Loss() 
    elif(args.lossfunc == 'KLDivLoss'): # 不可以
        criterion = nn.KLDivLoss() 
    elif(args.lossfunc == 'BCEWithLogitsLoss'):
        criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss损失函数，不能使preds趋近于[0, 1]区间中，训练过程趋向[-5, 9]    

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters()) # Adam默认lr是1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.001) # weight_decay导致训练效果极差
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1) # 每step_size, lr = lr * gamma
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True) # 自适应调整lr
    
    # 记录全部训练过程中的信息
    train_avgloss_epoch_list = []
    valid_avgloss_epoch_list = []
    # valid_avgqerror_epoch_list = [] # 每个epoch平均qerror
    valid_qerror_epoch_list = [] # 每个epoch所有qerror列表 (不包含inf和None)
    # 训练过程图片输出路径
    train_fig_file = model_path / f"{global_table.version}_{workload}-{model.name()}_loss{args.lossfunc}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}.png"
    # 训练过程数据输出路径
    train_log_file = model_path / f"{global_table.version}_{workload}-{model.name()}_loss{args.lossfunc}_ep{args.epochs}_bs{args.bs}_{args.train_num//1000}k-{seed}-log.pkl"
    
    # 初始化
    valid_time = 0
    best_valid_loss = float('inf')
    
    start_stmp = time.time()
    for epoch in range(args.epochs):
        global_epoch = epoch
        
        train_loss_iter_list = [] # 单个epoch内，每个iteration的loss
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
            
            train_loss_iter_list.append(loss.item())  
            
        avg_train_loss = sum(train_loss_iter_list) / len(train_loss_iter_list) # 单个epoch内，平均训练损失
        train_avgloss_epoch_list.append(avg_train_loss)
        dur_min = (time.time() - start_stmp) / 60
        L.info(f"Epoch {epoch+1}, loss: {avg_train_loss}, time since start: {dur_min:.1f} mins, lr: {optimizer.param_groups[0]['lr']}")
        # 在每个epoch结束时更新学习率【scheduler = StepLR 时使用】
        # scheduler.step()

        L.info(f"Test on valid set...")
        valid_stmp = time.time()
        model.eval()
        val_loss_iter_list = [] # 单个epoch内，每个iteration的loss
        val_q_error_iter_list = [] # 单个epoch内，每个iteration的中每个query的q_error

        inputs_iter_list, preds_iter_list, truecards_iter_list, collist_iter_list = [], [], [], [] # 单个epoch内所有inputs
        for _, data in enumerate(valid_loader):
            inputs, labels, truecards, collist = data
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()
            
            with torch.no_grad():
                preds = model(inputs)
                val_loss = criterion(preds, labels)
                val_loss_iter_list.append(val_loss.item())
                # 计算每个inputs在preds上的cardinality estimation与truecards的差（q-error）
                if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
                    inputs_iter_list.append(inputs)
                    preds_iter_list.append(preds)
                    truecards_iter_list.append(truecards)
                    collist_iter_list += collist
                    
        avg_val_loss = sum(val_loss_iter_list) / len(val_loss_iter_list)
        valid_avgloss_epoch_list.append(avg_val_loss)
        
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            all_inputs = torch.cat(inputs_iter_list, dim=0)
            all_preds = torch.cat(preds_iter_list, dim=0)
            all_truecards = torch.cat(truecards_iter_list, dim=0)
            all_collist = collist_iter_list

            start_decodeT = time.time()
            # 单线程
            # val_q_error_iter_list = decodePreds(inputs, preds, truecards, collist)
            # 多线程
            # val_q_error_iter_list = decodePreds_parallel(inputs, preds, truecards, collist)
            # 多进程
            val_q_error_iter_list, estimate_list = decodePreds_pool(all_inputs, all_preds, all_truecards, all_collist)
            L.info(f"Decode Time: {(time.time()-start_decodeT)}s")
            
            Q_error_print(val_q_error_iter_list)
            avg_q_error = sum([value.item() for value in val_q_error_iter_list if value is not None and value.item() != float('inf')]) / len(val_q_error_iter_list)
            valid_qerror_epoch_list.append([value.item() for value in val_q_error_iter_list if value is not None and value.item() != float('inf')])
        
        L.info(f"Validation Loss: {avg_val_loss}")
        # 自适应调整lr【scheduler = ReduceLROnPlateau时使用】
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_valid_loss:
            L.info('***** best valid loss for now!')
            best_valid_loss = avg_val_loss
            state['model_state_dict'] = model.state_dict()
            state['optimizer_state_dict'] = optimizer.state_dict()
            state['valid_error'] = {workload: avg_q_error}
            state['train_time'] = (valid_stmp-start_stmp-valid_time) / 60
            state['current_epoch'] = epoch
            torch.save(state, model_file)

        valid_time += time.time() - valid_stmp

    L.info(f"Training finished! Time spent since start: {(time.time()-start_stmp)/60:.2f} mins")
    L.info(f"Model saved to {model_file}, best valid: {state['valid_error']}, best valid loss: {best_valid_loss}")

    # 保存输出文件
    data_dict = {
        'train_avgloss_epoch_list': train_avgloss_epoch_list,
        'valid_avgloss_epoch_list': valid_avgloss_epoch_list,
        'valid_qerror_epoch_list': valid_qerror_epoch_list
        }
    with open(train_log_file, 'wb') as file:
        pickle.dump(data_dict, file)
    L.info(f"Log saved to {train_log_file}")
        
    
    '''
    画图 
        train_avgloss_epoch_list 和 valid_avgloss_epoch_list 使用左轴, 折线图
        valid_qerror_list 使用右轴, 箱型图
    '''
    fig, ax1 = plt.subplots()
    ax1.plot(train_avgloss_epoch_list, label='Training Loss', color='blue')
    ax1.plot(valid_avgloss_epoch_list, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left') # 在左轴上添加图例
    ax2 = ax1.twinx() # 在同一个图表上创建一个新的y轴，与原始的y轴（ax1）共享x轴。
    ax2.boxplot(valid_qerror_epoch_list, positions=[1]+[i * 10 for i in range(1, len(valid_qerror_epoch_list))], sym='+', vert=True, widths=6, showfliers=False)
    ax2.set_ylabel('Validation QError', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(10**-0.5, 10**2)  # 设置右轴的纵轴范围
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    plt.title('Training Loss, Validation Loss, and Validation QError Over Epochs')
    plt.savefig(train_fig_file)


'''
模型测试
'''
def test_lstm(dataset: str, version: str, workload: str, params:Dict[str, Any], overwrite: bool):
    global global_table
    global global_cols_alldomain
    global global_epoch
    
    torch.set_num_threads(NUM_THREADS)
    assert NUM_THREADS == torch.get_num_threads(), torch.get_num_threads()
    L.info(f"Torch threads: {torch.get_num_threads()}")
    
    model_file = MODEL_ROOT / dataset / f"{params['model']}.pt"
    L.info(f"Load model from {model_file} ...")
    state = torch.load(model_file, map_location=DEVICE)
    args = state['args']
    
    result_path = RESULT_ROOT / f"{dataset}"
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = result_path / f"{version}-{workload}-lstm.csv"
    
    global_epoch = args.epochs
    
    model = TextSentimentModel(args.hid_units).to(DEVICE)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    
    global_table = load_table(dataset, state['version'])
    Dataset, global_cols_alldomain = load_lstm_dataset(global_table, workload)
    test_dataset = make_dataset(Dataset['test'], int(args.train_num/10))
    L.info(f"Number of testing samples: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.bs, num_workers=NUM_THREADS)

    inputs_iter_list, preds_iter_list, truecards_iter_list, collist_iter_list = [], [], [], [] # 单个epoch内所有inputs
    for _, data in enumerate(test_loader):
        inputs, labels, truecards, collist = data
        inputs = inputs.to(DEVICE).float()
        labels = labels.to(DEVICE).float()
        
        with torch.no_grad():
            preds = model(inputs)
            inputs_iter_list.append(inputs)
            preds_iter_list.append(preds)
            truecards_iter_list.append(truecards)
            collist_iter_list += collist
    
    
    all_inputs = torch.cat(inputs_iter_list, dim=0)
    all_preds = torch.cat(preds_iter_list, dim=0)
    all_truecards = torch.cat(truecards_iter_list, dim=0)
    all_collist = collist_iter_list

    start_decodeT = time.time()
    val_q_error_iter_list, estimate_list = decodePreds_pool(all_inputs, all_preds, all_truecards, all_collist)
    L.info(f"Decode Time: {(time.time()-start_decodeT)}s")
    
    Q_error_print(val_q_error_iter_list)
    
    all_truecards_ = all_truecards[:, -1]
    with open(result_file, 'w') as f:
        writer = csv.writer(f)
        header = ["id", "True Card", "Estimate Card", "Validation Error", "Column List"]
        writer.writerow(header)
        
        for _ in range(len(all_truecards_)):
            row = [_, all_truecards_[_].item(), estimate_list[_].item(), val_q_error_iter_list[_].item(), all_collist[_]]
            writer.writerow(row)
    
    L.info(f"Test finished, result in {result_file}")
        
