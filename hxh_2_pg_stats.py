import pandas as pd

# 读取CSV文件
df = pd.read_csv('hxh_2_pg_stats.csv')

# 计算most_common_vals列每行的数组长度
df['most_common_vals_num'] = df['most_common_vals'].apply(lambda x: len(str(x).replace("{", "").replace("}", "").split(',')))
# df['most_common_freqs_sum'] = df['most_common_freqs'].apply(lambda x: sum(str(x).split(',')))
df['most_common_freqs_sum'] = df['most_common_freqs'].apply(lambda x: round(sum(map(float, str(x).replace("{", "").replace("}", "").split(','))), 15) if pd.notna(x) else 0)
# df['most_common_freqs_sum'] = df['most_common_freqs'].apply(lambda x: sum(map(float, str(x).replace("{", "").replace("}", "").split(','))) if pd.notna(x) else 0)
# 保存更改后的数据框到原文件
df.to_csv('hxh_2_pg_stats.csv', index=False)

# 打印处理后的数据框
print(df)
