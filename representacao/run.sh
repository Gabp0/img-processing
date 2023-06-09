#!/bin/bash

ENDI=$1

for i in $(seq 5 $ENDI)
do
    echo "Size = $i x $i"
    python3 digits.py features/$i_$i.txt $i $i > /dev/null
    python3 knn.py features/$i_$i.txt
done