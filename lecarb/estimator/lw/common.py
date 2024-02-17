import numpy as np
import pickle
import logging

from ..postgres import Postgres
from ...workload.workload import load_queryset, load_labels, query_2_sqls, query_2_vector
from ...constants import DATA_ROOT, PKL_PROTO

L = logging.getLogger(__name__)

# selectivity_list (np.array): selectivity for each attribute
def AVI(sel_list):
    return np.prod(sel_list) if len(sel_list) > 0 else 1.0

def EBO(sel_list):
    s = 1.0
    sorted_slist = np.sort(sel_list)
    for i in range(min(4, sel_list.size)):
        s = s * np.power(sorted_slist[i], 1 / (i+1))
    return s

def MinSel(sel_list):
    return sel_list.min() if len(sel_list) > 0 else 1.0

def encode_query(table, query, pg_est):
        range_features = query_2_vector(query, table, upper=1000) # 获取query的[]range表示
        sqls = query_2_sqls(query, table) # 获取query的sql表示
        sel_list = []
        for sql in sqls:
            pred, _ = pg_est.query_sql(sql)
            sel_list.append(pred / table.row_num)
        sel_list = np.array(sel_list)
        ce_features = np.round(np.array([AVI(sel_list), EBO(sel_list), MinSel(sel_list)]) * table.row_num)

        return np.concatenate([range_features, encode_label(ce_features)])

def encode_label(label):
    # +1 before log2 to deal with ground truth = 0 scenario
    return np.log2(label + 1)

def decode_label(label):
    return np.power(2, label) - 1

def encode_queries(table, queryset, labels, pg_est):
    X = []
    y = []
    gt = []

    for query, label in zip(queryset, labels):
        features = encode_query(table, query, pg_est)
        log2l = encode_label(label.cardinality)
        X.append(features)
        y.append(log2l)
        gt.append(label.cardinality)

    return np.array(X), np.array(y), np.array(gt)

def load_lw_dataset(table, workload, seed, bins):
    query_path = DATA_ROOT / table.dataset / "lw"
    query_path.mkdir(exist_ok=True)

    file_path = query_path / f"{table.version}_{workload}_{bins}_{seed}.pkl"
    if file_path.is_file():
        L.info(f"features already built in file {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # pg_est是Postgres的实例
    pg_est = Postgres(table, bins, seed)
    L.info(f"Start loading queryset:{workload} and labels for version {table.version} of dataset {table.dataset}...")
    queryset = load_queryset(table.dataset, workload)
    labels = load_labels(table.dataset, table.version, workload)

    '''
    w_dataset <dict> keys: train, valid, test
    lw_dataset['train'] 包含3个长度为100000的array: X【[range_features, encode_label(ce_features)]】, y【log后的基数】, gt【真实基数】

    X train len(tu_) 100000
    [   0.         1000.          500.          500.            0.
    1000.            0.         1000.          666.66668653  666.66668653
        0.         1000.          200.00000298  200.00000298    0.
    1000.         1000.         1000.            0.           93.03035587
        0.         1000.            0.         1000.          951.21949911
    951.21949911    9.17242751   12.14625045   13.61930296]
    
    y train len(tu_) 100000
    11.38586240064146
    
    gt train len(tu_) 100000
    2675
    '''
    lw_dataset = {}
    for group in queryset.keys(): # group: train, valid, test
        L.info(f"Start encode group: {group} with {len(labels[group])} queries...")
        lw_dataset[group] = encode_queries(table, queryset[group], labels[group], pg_est)

    with open(file_path, 'wb') as f:
        pickle.dump(lw_dataset, f, protocol=PKL_PROTO)

    return lw_dataset
