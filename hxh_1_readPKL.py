import pickle

# 指定.pkl文件的路径
# file_path = 'data/census13/original.pkl'
# file_path = 'data/census13/workload/base.pkl'
file_path = 'data/census13/workload/base-original-label.pkl'
# 打开文件并读取数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# print("data type is", type(data))  # <class 'dict'> dict_keys(['train', 'valid', 'test'])
# # print("data[\"train\"] type is", type(data["train"]))  # <class 'list'> 100000
# # # 现在，变量"data"包含了.pkl文件中的数据
# print(list(data.keys())[:1])
# print(len(data["train"]))

for key_ in data.keys():
    print(key_, "len is", len(data[key_]))
