#!/bin/bash

echo "Testing GPU"

dsets=(arxiv collab citation ddi protein ppa reddit.dgl products youtube amazon_cogdl yelp wikikg2 am)
filename=output_GPU_$(date +"%H_%M_%S_%m_%d").log

echo Log saved to $filename

for j in `seq 0 $((${#dsets[@]}-1))`;
do
    echo ${dsets[j]}
    GTEST_FILTER="SpMMTest.*" ../build/test/unit_tests --dataset ${dsets[j]}   --len 256 --datadir ~/PA3/data/  2>&1 | tee -a $filename 
done

