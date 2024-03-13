## data/census13/workload/base.pkl

train len is 100000; valid len is 10000; test len is 10000

```python
print("data type is", type(data))  # <class 'dict'> dict_keys(['train', 'valid', 'test'])
print("data[\"train\"] type is", type(data["train"]))  # <class 'list'> 100000
```

![Alt text](hxh_image/image_1.png)

## data/census13/workload/base-original-label.pkl
train len is 100000; valid len is 10000; test len is 10000

![Alt text](hxh_image/image_2.png)

## data/census13/original.table.pkl
['dataset',
 'version',
 'name',
 'data',
 'data_size_mb',
 'row_num',
 'col_num',
 'columns']

 ![Alt text](hxh_image/image_3.png)

 0311 是uniform的
 0312 是train和 test valid完全不相交的
 0312_1 是 if _ % 8 != 0: