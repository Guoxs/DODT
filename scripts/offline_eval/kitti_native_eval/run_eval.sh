#!/bin/bash

set -e

cd $1
echo "$3" | tee -a ./$4_results_$2.txt
./evaluate_object_3d_offline  /data/hk1/guoslu/Datasets/Kitti/tracking/training/label_2/ $2/$3 | tee -a ./$4_results_$2.txt

cp $4_results_$2.txt $5
