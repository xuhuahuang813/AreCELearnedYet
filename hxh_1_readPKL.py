import pickle

# 指定.pkl文件的路径
file_path = 'data/census13/original.pkl'

# 打开文件并读取数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 现在，变量"data"包含了.pkl文件中的数据
print(data)
