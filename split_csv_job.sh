#!/bin/bash

#$ -cwd -V
#$-l h_rt=12:59:59
#$ -pe smp 20
#$ -R y

echo "Running script on `hostname`"
echo "Start on `date`"

python split_train_test_csv.py

echo "End on `date`"
