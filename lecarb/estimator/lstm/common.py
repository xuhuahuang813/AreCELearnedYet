'''
hxh
lstm 特征编码
'''
import numpy as np
import pandas as pd
import pickle
import logging
import os
import csv
import time
import concurrent.futures
from itertools import product

from ..postgres import Postgres
from ...workload.workload import load_queryset, load_labels, query_2_vector_lstm
from ...workload.workload import Query
from ...dataset.dataset import Table
from ...constants import DATA_ROOT, PKL_PROTO
from ...dtypes import is_categorical


L = logging.getLogger(__name__)

# 字典，存储column字符串和对应的1w维向量。其中，列名间用"/"连接。
clos_vector = {}
# 字典，存储column字符串和对应的联合域值。字典会dump，用于Lstm模型得到preds的10000维向量后解码。其中，列名间用"/"连接。
clos_alldomain = {}

# AVI
def AVI(sel_list):
    return np.prod(sel_list) if len(sel_list) > 0 else 1.0

'''
# 分析单个query中每个谓词的选择率，返回所有谓词选择率乘积
def analyzeCorrSingle(table, query, pg_est):
    colNotNone = [col for col, pred in query.predicates.items() if pred is not None] # 存在谓词的列
    sqls = query_2_sqls(query, table) # 获取query的sql表示
    sel_list = []
    for sql in sqls:
        pred, _ = pg_est.query_sql(sql)
        sel_list.append(pred / table.row_num)
    sel_list = np.array(sel_list)
    return colNotNone, AVI(sel_list)*table.row_num

def analyzeCorr(corr_path, table, queryset, labels, pg_est):
    L.info(f"Start analyzing correlation between columns...")
    
    ana_start = time.time()
    # 获得AVI和真实基数的Qerror，用于发现存在较强相关性的列
    cloListAll, AVIListAll, lableListAll = [], [], []
    for group in queryset.keys():
        for _, (query, lable) in enumerate(zip(queryset[group], labels[group])):
            cloListAll_, AVIListAll_ = analyzeCorrSingle(table, query, pg_est)
            cloListAll.append(cloListAll_)
            AVIListAll.append(AVIListAll_)
            lableListAll.append(lable.cardinality)
            if _ % 1000 == 0:
                L.info(f"Finish analyzing {_}th query, time cost: {time.time() - ana_start}s")
    ana_end = time.time()
    L.info(f"Finish analyzing correlation between columns, time cost: {ana_end - ana_start}s")
    L.info(f"Start writing correlation into {corr_path}...")    
    # 将AVI和真实基数的Qerror写入corr_path文件
    if os.path.exists(corr_path):
        os.remove(corr_path)
    with open(corr_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['cloList', 'AVIList', 'lableList'])
        writer.writerows(zip(cloListAll, AVIListAll, lableListAll))
'''

'''
input: col_eachDomain
output: unionDomain
被encode_query_y()调用
'''

'''
这样做会导致很多超过10000。
因为每个单列的域值是[0, 1]，一旦第一个列取到域值的顶端，返回值就已经是10000。后面的列会使这个值大于10000。
例如，age列的阈值范围是min=17， max=90。当年龄为90的时候，所有返回值都是大于10000的
'''
# def cols_2_unionDomain(row, col_list, col_list_num: int, table: Table):
#     cols_eachDomain = []
#     for col_name in col_list:
#         # if(col_list_num == 3):
#         #     L.info(f"{col_name} is {row[col_name]}, {table.columns[col_name].value_2_domain[row[col_name]]}, {type(table.columns[col_name].value_2_domain[row[col_name]])} ")
#         cols_eachDomain.append(table.columns[col_name].value_2_domain[row[col_name]])
#     if col_list_num == 1:
#         return int(round(cols_eachDomain[0]*10000))
#     elif col_list_num == 2:
#         return int(round(cols_eachDomain[0]*10000)) + int(round(cols_eachDomain[1]*100))
#     elif col_list_num == 3:
#         return int(round(cols_eachDomain[0]*8000)) + int(round(cols_eachDomain[1]*400)) + int(round(cols_eachDomain[2]*20))
#     elif col_list_num == 4:
#         return int(round(cols_eachDomain[0]*10000)) + int(round(cols_eachDomain[1]*1000)) + int(round(cols_eachDomain[2]*100)) + int(round(cols_eachDomain[3]*10))
#     L.error("cols_2_unionDomain go wrong")
#     return None

'''
input: query
output: 1w维向量
'''    
# 编码y
def encode_query_y(query:Query, table: Table):
    # 获取非None列
    # TODO col_list最好在生成query的时候生成
    col_list = []
    # 获取非None列的个数
    col_list_num = query.ncols
    for col_, pre_ in query.predicates.items():
        if pre_ is not None:
            col_list.append(col_)
    col_list_str = "/".join(col_list)
    
    # 如果已经生成过，则直接返回col_list_str中对应的1w维
    if col_list_str in clos_vector:
        return col_list_str, clos_vector[col_list_str]
    else:
        # 当前这些列的组合没有生成过1w维
        # 按照列的顺序，排列所有行
        selected_data = table.data[col_list].reset_index(drop=True)
        df_sorted = selected_data.sort_values(by=col_list).reset_index(drop=True)
        # 使用 value_counts 统计每个组合的数量
        grouped_counts = df_sorted.groupby(col_list).size().reset_index(name='count')
        grouped_counts = grouped_counts[grouped_counts['count'] > 0]
        # 计算累计概率cumulative_probability
        # grouped_counts包含[col_list每列对应的值, count, cumulative_probability]
        grouped_counts['cumulative_probability'] = grouped_counts['count'].cumsum() / len(df_sorted)
        # 计算所有列的在1w中的域位置
        # grouped_counts['union_domain'] = grouped_counts.apply(cols_2_unionDomain, axis=1, col_list=col_list, col_list_num=col_list_num, table=table)
        
        # 根据数据类型获取unique_values
        # 属性列获取所有属性，int64列获取[min, max]之间的所有数值（递增1）
        unique_values = []
        for col in col_list:
            if is_categorical(table.columns[col].dtype):
                unique_values.append(pd.unique(grouped_counts[col].values))
            else:
                unique_values.append(range(table.columns[col].minval, table.columns[col].maxval + 1))
        # 笛卡尔积
        cartesian_product = list(product(*unique_values))
        result_df = pd.DataFrame(cartesian_product, columns=col_list)
        # 增加一列index_alldomain，内容为新的索引值
        result_df['index_alldomain'] = pd.Series(range(len(result_df))) / len(result_df)
        # 乘以10000后四舍五入取整
        result_df['index_alldomain'] = round(result_df['index_alldomain'] * 10000)
        # 排序新的DataFrame
        result_df.sort_values(by=col_list, inplace=True)
        result_df.reset_index(drop=True, inplace=True)
        # 合并原始的grouped_counts中的count和cumulative_probability列，只保留存在的行
        grouped_counts = pd.merge(result_df, grouped_counts, how='inner', on=col_list)
        # 保存到clos_alldomain
        for col in col_list:
            # result_df[col] = table.columns[col].value_2_domain[result_df[col]]
            result_df[col] = result_df[col].apply(lambda x: table.columns[col].value_2_domain[x])
        clos_alldomain[col_list_str] = result_df
        
        unionDomain = np.zeros(10000) # 1w维向量
        # 给unionDomain数组赋值
        last_cumulative_probability = 0.0
        last_union_domain = 0
        for _, row in grouped_counts.iterrows():
            new_union_domain = row["index_alldomain"]
            if(new_union_domain > 10000):
                L.error(f"union domain >= 10000, index_all domain is {new_union_domain}, \n col list is {col_list}, \n row is {row}")
            unionDomain[int(last_union_domain) : int(new_union_domain)] = last_cumulative_probability
            last_cumulative_probability = row["cumulative_probability"]
            last_union_domain = new_union_domain
        unionDomain[int(last_union_domain) : ] = 1.0
        # 将1w维向量放入字典，避免重复生成
        clos_vector[col_list_str] = unionDomain
        return col_list_str, clos_vector[col_list_str]


# 编码X
def encode_query_X(table:Table, query:Query, label):
    range_features = query_2_vector_lstm(query, table)
    card_feature = np.array([label/table.row_num])
    
    # 11 = 1(card) + 5(一个查询中最大谓词数量)*2(上下边界)
    padded_feature = np.zeros(11)

    # 将 card_feature 和 range_features 填充到零数组的相应位置
    padded_feature[0] = card_feature
    padded_feature[1:len(range_features)+1] = range_features

    return padded_feature

def encode_queries(table:Table, queryset, labels):
    X = [] #编码 [card/rowCount, lowerBound, upperBound, ...]
    y = [] #10000维向量
    card = [] # card
    colList = [] # 存储每50个query是对应的哪些列
    
    for i, (query, label) in enumerate(zip(queryset, labels), start=1):
        if i % 1000 == 0:
            L.info(f"i is {i}")
        X.append(encode_query_X(table, query, label.cardinality))
        card.append(label.cardinality)
        if i % 50 == 0:
            col_l, y_vector = encode_query_y(query, table)
            y.append(y_vector)
            colList.append(col_l)
    return X, y, card, colList
    

# load training dataset
def load_lstm_dataset(table:Table, workload, seed, bins):
    query_path = DATA_ROOT / table.dataset / "lstm"
    query_path.mkdir(exist_ok=True)
    
    # query的pkl存储路径。如果生成过则直接加载。e.g.data/census13/lw/original_base_200_123.pkl。
    file_path = query_path / f"{table.version}_{workload}_{bins}_{seed}.pkl"
    # cols_alldomain的pkl存储路径。data/census13/lstm/original_lstm-small_200_123_colsAllDomain.pkl
    colsAllDomain_path = query_path / f"{table.version}_{workload}_{bins}_{seed}_colsAllDomain.pkl"
    
    # TODO 记得反注释
    if file_path.is_file() and colsAllDomain_path.is_file():
        L.info(f"features already built in file {file_path}, domain of all columns already build in file {colsAllDomain_path}")
        with open(file_path, 'rb') as f1, open(colsAllDomain_path, 'rb') as f2:
            return pickle.load(f1), pickle.load(f2)
    
    L.info(f"Start loading queryset:{workload} and labels for version {table.version} of dataset {table.dataset}...")
    
    # 加载data/census13/workload/base.pkl
    queryset = load_queryset(table.dataset, workload)
    # 加载data/census13/workload/base-original-label.pkl
    labels = load_labels(table.dataset, table.version, workload)
    
    # # 使用pg计算AVI
    # analyzeCorr(corr_path, table, queryset, labels, pg_est)
    
    # 编码
    lw_dataset = {}
    for group in queryset.keys(): # train, valid, test
        L.info(f"Start encode group: {group} with {len(labels[group])} queries...")
        lw_dataset[group] = encode_queries(table, queryset[group], labels[group])
        
    # 保存query的pkl
    with open(file_path, 'wb') as f:
        pickle.dump(lw_dataset, f, protocol=PKL_PROTO)
    # 保存cols_alldomain的pkl
    with open(colsAllDomain_path, 'wb') as f:
        pickle.dump(clos_alldomain, f, protocol=PKL_PROTO)
    
        
    return lw_dataset, clos_alldomain