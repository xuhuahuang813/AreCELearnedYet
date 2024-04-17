## 开发环境设置

设置步骤：
* 安装 Just：
  * 在 MacOS 上：`brew install just`
  * 在 Linux 上：`curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin`
* 安装 Poetry：`pip install poetry`
* 安装 Python 依赖：`just install-dependencies`

在 `Justfile` 中定义了所有用到的命令，包括数据集处理、工作负载生成、模型训练及测试等。运行 `just -l` 可以查看支持的任务列表。

所有环境配置（例如数据路径、数据库配置）都在 `.env` 文件中设置。

## 模型训练和测试
* 训练lstm模型
```
just train-lstm census13 original merge1w 64_2048 200 10000 8 0 123 MSELoss
```
* 测试生成模型
```
# 其中original_merge1w-lstm_64_2048_lossMSELoss_ep200_bs8_10k-123需要根据实际情况替换为训练好的模型的名字
just test-lstm original_merge1w-lstm_64_2048_lossMSELoss_ep200_bs8_10k-123 census13 original 1w 123
```

## 论文和代码参考：
* AreCELearnedYet: 论文（https://www.vldb.org/pvldb/vol14/p1640-wang.pdf ），代码（https://github.com/sfu-db/AreCELearnedYet ）。 
* MSCN: 论文（https://arxiv.org/pdf/1809.00677.pdf ），代码（https://github.com/andreaskipf/learnedcardinalities ）。
* LW-NN: 论文（https://dl.acm.org/doi/pdf/10.14778/3329772.3329780 ），作者未开源代码。

