import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from typing_extensions import Protocol

import numpy as np
import pandas as pd
import math

from ..dtypes import is_categorical
from ..dataset.dataset import Table, Column
from .workload import Query, new_query

L = logging.getLogger(__name__)

# QuerySet = {'train':[], 'valid':[], 'test':[]}

"""====== Attribute Selection Functions ======"""

class AttributeSelFunc(Protocol):
    def __call__(self, table: Table, params: Dict[str, Any]) -> List[str]: ...

def asf_pred_number(table: Table, params: Dict[str, Any]) -> List[str]:
    if 'whitelist' in params:
        attr_domain = params['whitelist']
    else:
        blacklist = params.get('blacklist') or []
        attr_domain = [c for c in list(table.data.columns) if c not in blacklist]
    nums = params.get('nums')
    nums = nums or range(1, len(attr_domain)+1)
    num_pred = np.random.choice(nums)
    assert num_pred <= len(attr_domain)
    return np.random.choice(attr_domain, size=num_pred, replace=False)

def asf_comb(table: Table, params: Dict[str, Any]) -> List[str]:
    assert 'comb' in params and type(params['comb']) == list, params
    for c in params['comb']:
        assert c in table.columns, c
    return params['comb']

def asf_naru(table: Table, params: Dict[str, Any]) -> List[str]:
    num_filters = np.random.randint(5, 12)
    return np.random.choice(table.data.columns, size=num_filters, replace=False)

"""====== Center Selection Functions ======"""

class CenterSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]: ...

DOMAIN_CACHE = {}
# This domain version makes sure that query's cardinality > 0
def csf_domain(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    global DOMAIN_CACHE
    key = tuple(sorted(attrs))
    if key not in DOMAIN_CACHE:
        data_from = params.get('data_from') or 0
        DOMAIN_CACHE[key] = table.data[data_from:][attrs].drop_duplicates().index
        assert len(DOMAIN_CACHE[key]) > 0, key
    #  L.debug(f'Cache size: {len(DOMAIN_CACHE)}')
    row_id = np.random.choice(DOMAIN_CACHE[key])
    return [table.data.at[row_id, a] for a in attrs]

ROW_CACHE = None
GLOBAL_COUNTER = 1000
def csf_distribution(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    global GLOBAL_COUNTER
    global ROW_CACHE
    if GLOBAL_COUNTER >= 1000:
        data_from = params.get('data_from') or 0
        ROW_CACHE = np.random.choice(range(data_from, len(table.data)), size=1000)
        GLOBAL_COUNTER = 0
    row_id = ROW_CACHE[GLOBAL_COUNTER]
    GLOBAL_COUNTER += 1
    #  data_from = params.get('data_from') or 0
    #  row_id = np.random.choice(range(data_from, len(table.data)))
    return [table.data.at[row_id, a] for a in attrs]

def csf_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_ids = np.random.choice(len(table.data), len(attrs))
    return [table.data.at[i, a] for i, a in zip(row_ids, attrs)]

def csf_vocab_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    centers = []
    for a in attrs:
        col = table.columns[a]
        centers.append(np.random.choice(col.vocab))
    return centers

def csf_domain_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    centers = []
    for a in attrs:
        col = table.columns[a]
        if is_categorical(col.dtype): # randomly pick one point from domain for categorical
            centers.append(np.random.choice(col.vocab))
        else: # uniformly pick one point from domain for numerical
            centers.append(random.uniform(col.minval, col.maxval))
    return centers

def csf_naru(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_id = np.random.randint(0, len(table.data))
    return [table.data.at[row_id, a] for a in attrs]

def csf_naru_ood(table: Table, attrs: List[str], params: Dict[str, Any]) -> List[Any]:
    row_ids = np.random.choice(len(table.data), len(attrs))
    return [table.data.at[i, a] for i, a in zip(row_ids, attrs)]

"""====== Width Selection Functions ======"""

class WidthSelFunc(Protocol):
    def __call__(self, table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query: ...

def parse_range(col: Column, left: Any, right: Any) -> Optional[Tuple[str, Any]]:
    #  if left <= col.minval and right >= col.maxval:
    #      return None
    #  if left == right:
    #      return ('=', left)
    if left <= col.minval:
        return ('<=', right)
    if right >= col.maxval:
        return ('>=', left)
    return ('[]', (left, right))

def wsf_uniform(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        # NaN/NaT literal can only be assigned to = operator
        if pd.isnull(c) or is_categorical(table.columns[a].dtype):
            query.predicates[a] = ('=', c)
            continue
        col = table.columns[a]
        
        # width = random.uniform(0, col.maxval-col.minval)
        # 保证width是范围内的偶数，这样可以保证query中都是整数
        width = random.randint(0, col.maxval-col.minval)
        if width % 2 != 0:
            width += 1
            
        query.predicates[a] = parse_range(col, c-width/2, c+width/2)
    return query

def wsf_exponential(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        # NaN/NaT literal can only be assigned to = operator
        if pd.isnull(c) or is_categorical(table.columns[a].dtype):
            query.predicates[a] = ('=', c)
            continue
        col = table.columns[a]
        lmd = 1 / ((col.maxval - col.minval) / 10)
        width = random.expovariate(lmd)
        query.predicates[a] = parse_range(col, c-width/2, c+width/2)
    return query

def wsf_naru(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    ops = np.random.choice(['>=', '<=', '='], size=len(attrs))
    for a, c, o in zip(attrs, centers, ops):
        if table.columns[a].vocab_size >= 10:
            query.predicates[a] = (o, c)
        else:
            query.predicates[a] = ('=', c)
    return query

def wsf_equal(table: Table, attrs: List[str], centers: List[Any], params: Dict[str, Any]) -> Query:
    query = new_query(table, ncols=len(attrs))
    for a, c in zip(attrs, centers):
        query.predicates[a] = ('=', c)
    return query

class QueryGenerator(object):
    table: Table
    attr: Dict[AttributeSelFunc, float]
    center: Dict[CenterSelFunc, float]
    width: Dict[WidthSelFunc, float]
    attr_params: Dict[str, Any]
    center_params: Dict[str, Any]
    width_params: Dict[str, Any]

    def __init__(
            self, table: Table,
            attr: Dict[AttributeSelFunc, float],
            center: Dict[CenterSelFunc, float],
            width: Dict[WidthSelFunc, float],
            attr_params: Dict[str, Any],
            center_params: Dict[str, Any],
            width_params: Dict[str, Any]
            ) -> None:
        self.table = table
        self.attr = attr
        self.center = center
        self.width = width
        self.attr_params = attr_params
        self.center_params = center_params
        self.width_params = width_params

    # 返回一个query
    def generate(self) -> Query:
        attr_func = np.random.choice(list(self.attr.keys()), p=list(self.attr.values()))
        #  L.info(f'start generate attr {attr_func.__name__}')
        attr_lst = attr_func(self.table, self.attr_params)

        center_func = np.random.choice(list(self.center.keys()), p=list(self.center.values()))
        #  L.info(f'start generate center points {center_func.__name__}')
        center_lst = center_func(self.table, attr_lst, self.center_params)

        width_func = np.random.choice(list(self.width.keys()), p=list(self.width.values()))
        #  L.info(f'start generate widths {width_func.__name__}')
        return width_func(self.table, attr_lst, center_lst, self.width_params)
    
    # hxh 生成lstm数据集
    # 返回一组query。组中query个数等于单个序列长度。
    def generate_lstm(self, queryNumPerSeq:int, group:str) -> List[Query]:
        # global QuerySet
        
        # num_domain_left = [0, 0, 1, 1, 1, 1]
        # # num_domain_right = [0, 10000, 100, 21, 10, 10] # 包含"capital_gain"列，min=0, max=99999, vocab size=123 【这个列会导致query解码时间很长】
        # num_domain_right = [0, 100, 100, 21, 10, 10]

        queries = []
        
        # 选择列
        # num_pred = np.random.randint(1, 4) # 单个查询中包含列的个数
        # attr_domain = [c for c in list(self.table.data.columns) if self.table.columns[c].vocab_size > num_domain_left[num_pred] and self.table.columns[c].vocab_size <= num_domain_right[num_pred] and c != "capital_loss"]
        # attr_lst = np.random.choice(attr_domain, size=num_pred, replace=False)
        
        allAttrDomain = ['race', 'education', 'marital_status', 'workclass', 'occupation', 'education_num', 'relationship', 'hours_per_week', 'sex', 'age', 'native_country',
                        'marital_status/race', 'education_num/race', 'workclass/marital_status', 'education/sex', 'education_num/marital_status', 'occupation/native_country', 
                        'age/hours_per_week', 'race/sex', 'marital_status/sex', 'sex/native_country', 'workclass/sex', 'education_num/sex', 'education/hours_per_week', 
                        'age/education_num', 'race/hours_per_week', 'marital_status/hours_per_week', 'occupation/race', 'workclass/hours_per_week', 'education_num/hours_per_week', 
                        'age/native_country', 'workclass/education', 'education/education_num', 'age/occupation', 'occupation/sex', 'education/native_country', 'workclass/education_num', 
                        'race/native_country', 'education/occupation', 'marital_status/native_country', 'age/race', 'workclass/native_country', 'education_num/native_country', 
                        'occupation/hours_per_week', 'age/workclass', 'marital_status/occupation', 'workclass/occupation', 'education/race', 'sex/hours_per_week', 'education_num/occupation', 
                        'hours_per_week/native_country', 'age/sex', 'education/marital_status', 'workclass/race',
                        'workclass/marital_status/sex', 'workclass/education/sex', 'workclass/education_num/sex', 'education_num/marital_status/sex', 'workclass/race/sex', 
                        'education/marital_status/race', 'education/education_num/race', 'workclass/education/education_num', 'education/education_num/marital_status', 'marital_status/race/sex', 
                        'education/marital_status/sex', 'education/education_num/sex', 'workclass/marital_status/race', 'education/race/sex', 'workclass/education/race', 'workclass/education_num/race', 
                        'workclass/education/marital_status',
                        'workclass/marital_status/race/sex']
        
        attr_domain_prob_norm = [0.004355113742238902, 0.007502581586102339, 0.005265602335970787, 0.005945655236286631, 0.007327941360382217, 0.007502581586102339, 0.0048484730146689, 0.012351054600771239, 0.0018756453965255847, 0.01164673262018224, 0.010114075350639688, 0.009620716078209688, 0.01185769532834124, 0.011211257572257417, 0.009378226982627923, 0.012768183922073125, 0.017442016711021906, 0.023997787220953474, 0.0062307591387644875, 0.007141247732496371, 0.01198972074716527, 0.007821300632812214, 0.009378226982627923, 0.019853636186873577, 0.019149314206284573, 0.01670616834301014, 0.017616656936742026, 0.011683055102621118, 0.01829670983705787, 0.019853636186873577, 0.021760807970821923, 0.01344823682238897, 0.015005163172204678, 0.018974673980564457, 0.009203586756907802, 0.017616656936742026, 0.01344823682238897, 0.014469189092878589, 0.014830522946484556, 0.015379677686610471, 0.016001846362421142, 0.016059730586926318, 0.017616656936742026, 0.019678995961153457, 0.01759238785646887, 0.012593543696353003, 0.01327359659666885, 0.01185769532834124, 0.014226699997296822, 0.014830522946484556, 0.022465129951410923, 0.01352237801670782, 0.012768183922073125, 0.010300768978525533, 0.013086902968783003, 0.015323882218914554, 0.015323882218914554, 0.014643829318598709, 0.012176414375051117, 0.017123297664312028, 0.01936027691444358, 0.02095081840849131, 0.020270765508175463, 0.011496361474735275, 0.014643829318598709, 0.01688080856873026, 0.01556637131449632, 0.013733340724866825, 0.01780335056462787, 0.01780335056462787, 0.01871383915835976, 0.017442016711021906]
        
        attr_lst = np.random.choice(allAttrDomain, size=1, replace=False, p=attr_domain_prob_norm)[0].split("/")
        # attr_lst = allAttrDomain[num % len(allAttrDomain)].split("/")
        while len(queries) < queryNumPerSeq: 
            # L.info(f"Start generate queries of one seq. queryNumPerSeq is {queryNumPerSeq}")
            
            center_func = np.random.choice(list(self.center.keys()), p=list(self.center.values()))
            #  L.info(f'start generate center points {center_func.__name__}')
            center_lst = center_func(self.table, attr_lst, self.center_params)

            width_func = np.random.choice(list(self.width.keys()), p=list(self.width.values()))
            #  L.info(f'start generate widths {width_func.__name__}')
            
            query_ = width_func(self.table, attr_lst, center_lst, self.width_params)
            queries.append(query_)
            
            # try_num = 1
            # while len(queries) == queryNumPerSeq -1:
            #     if group == 'train':
            #         if (query_ not in QuerySet['train'] or attr_lst == ['sex'] or attr_lst == ['race'] or try_num >= 30000):
            #             queries.append(query_)
            #             QuerySet['train'].append(query_)
            #             break
            #         else:
            #             try_num += 1
            #             if(try_num % 10000 == 0):  
            #                 L.info(f"Query has already generated. {attr_lst} \n {query_}")
            #             center_lst = center_func(self.table, attr_lst, self.center_params)
            #             query_ = width_func(self.table, attr_lst, center_lst, self.width_params)
            #     elif group == 'valid' or group == 'test':
            #         if ((query_ not in QuerySet['train'] and  query_ not in QuerySet[group]) or attr_lst == ['sex'] or attr_lst == ['race'] or try_num >= 100000):
            #             queries.append(query_)
            #             QuerySet[group].append(query_)
            #             break
            #         else:
            #             try_num += 1
            #             if(try_num % 10000 == 0):  
            #                 L.info(f"Query has already generated. {attr_lst} \n {query_}")
            #             center_lst = center_func(self.table, attr_lst, self.center_params)
            #             query_ = width_func(self.table, attr_lst, center_lst, self.width_params)
            #     else:
            #         L.error("Wrong in generate queries")
        return queries