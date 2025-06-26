#!/bin/bash

for eps in 5 7 10 12
do
  echo "Running TEM with epsilon $eps"
  python pds.py --epsilon $eps --model tem
done
