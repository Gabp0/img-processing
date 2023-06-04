#!/bin/bash

# varies the distance between 1 to 100
for i in {1..100}
do
    echo "Distance: $i"
    python3 cartas.py -l $i | grep "Cartas corretas"
done