#!/bin/bash

datasets=('census13')
versions=('original')
workloads=('merge1w')

hid_units=('64_64_64')

bins=('200')
train_nums=('10000')
bss=('32')
sizelimits=('0')
seeds=('123')

tips=('')

for dataset in "${datasets[@]}"; do
  for version in "${versions[@]}"; do
    for workload in "${workloads[@]}"; do
      for hid_unit in "${hid_units[@]}"; do
        for bin in "${bins[@]}"; do
          for train_num in "${train_nums[@]}"; do
            for bs in "${bss[@]}"; do
              for sizelimit in "${sizelimits[@]}"; do
                for seed in "${seeds[@]}"; do
                    output_file="./hxh_log/lwnn_${dataset}_${version}_${workload}_${hid_unit}_trainnum${train_num}_bs${bs}_seed${seed}_output.log"
                    
                    echo "Output file: $output_file"
                    
                    nohup just train-lw-nn $dataset $version $workload $hid_unit $bin $train_num $bs $sizelimit $seed> $output_file &
                done
              done
            done
          done
        done
      done
    done
  done
done
