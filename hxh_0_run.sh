#!/bin/bash

# nohup ./hxh_0_run.sh &
# poetry run python -m lecarb train -s123 -dcensus13 -voriginal -wlstm-small -elstm --params "{'epochs': 100, 'bins': 200, 'hid_units': '256_512_1024_2048_4096_4096', 'train_num': 1000, 'bs': 32}" --sizelimit 0

# rm -rf output/model/census13/*
# rm -rf hxh_log/*

# dataset='census13'
# version='original'
# workload='lstm-small'
# # hid_units='256_512_1024_2048_4096_8192'
# hid_units='256_512_1024_2048_4096_4096'
# bins='200'
# train_num='1000'
# bs='32'
# sizelimit='0'
# seed='123'
# lossfunc='MSELoss'

# tips=''

# output_file="./hxh_log/${dataset}_${version}_${workload}_${hid_units}_trainnum${train_num}_bs${bs}_seed${seed}_lossfuc${lossfunc}_${tips}_output.log"

# echo "Output file: $output_file"

# nohup just train-lstm $dataset $version $workload $hid_units $bins $train_num $bs $sizelimit $seed $lossfunc > $output_file &

#!/bin/bash

datasets=('census13')
versions=('original')
workloads=('lstm-1w')

hid_units=('256_1024_4096')
bins=('200')
train_nums=('10000')
bss=('32')
sizelimits=('0')
seeds=('123')
lossfuncs=('MSELoss')

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
                  for lossfunc in "${lossfuncs[@]}"; do
                    for tip in "${tips[@]}"; do
                      output_file="./hxh_log/${dataset}_${version}_${workload}_${hid_unit}_trainnum${train_num}_bs${bs}_seed${seed}_lossfuc${lossfunc}_${tip}_output.log"
                      
                      echo "Output file: $output_file"
                      
                      nohup just train-lstm $dataset $version $workload $hid_unit $bin $train_num $bs $sizelimit $seed $lossfunc > $output_file &
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
