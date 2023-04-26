#!/bin/bash
for i in $(seq 1 7)
do 
  py floresta.py imgs/f$i.png out$i.png 
done
