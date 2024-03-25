#!/bin/bash

datasets=('census13')
versions=('original+original_ind_0.1' 'original+original_ind_0.2' 'original+original_ind_0.3' 'original+original_ind_0.4' 'original+original_ind_0.5')  # 添加了第二个版本
workloads=('lstm-2k-IND0.1' 'lstm-2k-IND0.2' 'lstm-2k-IND0.3' 'lstm-2k-IND0.4' 'lstm-2k-IND0.5')  # 添加了第二个工作负载

hid_units=('64_2048')
bins=('200')
train_nums=('10000')
bss=('8')
sizelimits=('0')
seeds=('123')
lossfuncs=('MSELoss')
tips=('')

for dataset in "${datasets[@]}"; do
  for i in "${!versions[@]}"; do  # 使用索引迭代版本
    version="${versions[$i]}"
    workload="${workloads[$i]}"
    for hid_unit in "${hid_units[@]}"; do
      for bin in "${bins[@]}"; do
        for train_num in "${train_nums[@]}"; do
          for bs in "${bss[@]}"; do
            for sizelimit in "${sizelimits[@]}"; do
              for seed in "${seeds[@]}"; do
                for lossfunc in "${lossfuncs[@]}"; do
                  for tip in "${tips[@]}"; do
                    output_file="./hxh_log/${dataset}_${version}_${workload}_${hid_unit}_trainnum${train_num}_bs${bs}_seed${seed}_lossfuc${lossfunc}_${tip}_output.log"
                    
                    echo "Output file: $output_file"
                    
                    nohup just train-lstm $dataset $version $workload $hid_unit $bin $train_num $bs $sizelimit $seed $lossfunc > $output_file &
                    # Break out of the innermost loop after the first iteration
                    break
                  done
                  # Break out of the outermost loop after the first iteration
                  break 2
                done
              done
            done
          done
        done
      done
    done
  done
done
