#!/bin/bash

#$ -cwd -V
#�$-l h_rt=3:59:59
#$ -pe smp 20
#$ -R y

echo "running on `host` at `date`"
python generate_csv.py
echo "end on `date`"
