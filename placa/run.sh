#!/bin/bash
for i in $(seq 1 15)
do 
  python3 placa.py imgs/$i.jpg out.png 
done
