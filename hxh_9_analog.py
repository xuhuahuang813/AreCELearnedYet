

# 指定日志文件路径
# file_path = 'hxh_log_03062111/好-census13_original_lstm-1wAllDomain_64_1024_trainnum10000_bs8_seed123_lossfucMSELoss__output.log'
# file_path = 'hxh_log_03062111/好-census13_original_lstm-1wAllDomain_64_2048_trainnum10000_bs16_seed123_lossfucMSELoss__output.log'
file_path = 'hxh_log_03062400/好census13_original_lstm-1wAD_256_2048_trainnum10000_bs8_seed123_lossfucMSELoss__output.log'

def split_log_by_decode_time(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    log_segments = []
    current_segment = []

    for line in lines:
        current_segment.append(line)
        if 'Decode Time' in line:
            log_segments.append(current_segment)
            current_segment = []

    return log_segments

def extract_specific_lines(log_segment):
    inf_coll_lines = [line for line in log_segment if line.startswith('【inf】【coll】')]
    two_coll_lines = [line for line in log_segment if line.startswith('【2】【coll】')]

    return inf_coll_lines, two_coll_lines


log_segments = split_log_by_decode_time(file_path)

dic_inf, dic_2 = {}, {}

for segment in log_segments[-5:]:
    inf_coll_lines, two_coll_lines = extract_specific_lines(segment)

    print(f'Inf Coll Lines in Segment:')
    for line in inf_coll_lines:
        domain = line.strip().replace("【inf】【coll】", "")
        dic_inf[domain] = dic_inf.get(domain, 0) + 1

    for line in two_coll_lines:
        line = line.strip().replace("【2】【coll】", "").split("【Q】")
        domain, qerr = line[0], float(line[1])
        dic_2.setdefault(domain, []).append(qerr)

    
    # 打印 dic_inf 字典按值从大到小排序
    sorted_inf_items = sorted(dic_inf.items(), key=lambda x: x[1], reverse=True)
    for item in sorted_inf_items:
        print(f'Domain: {item[0]}, Count: {item[1]}')

    # 打印 dic_2 字典按值的列表和从大到小排序
    # sorted_2_items = sorted(dic_2.items(), key=lambda x: sum(map(len, x[1])), reverse=True)
    sorted_2_items = sorted(dic_2.items(), key=lambda x: sum(map(float, x[1])), reverse=True)

    print('\n Sorted Dictionary for [2][coll] (by sum of list lengths):')
    for item in sorted_2_items:
        print(f'\n Domain: {item[0]}, Count: {sorted(item[1])}')
    
    print('-' * 50)


