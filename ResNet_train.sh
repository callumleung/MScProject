#!/bin/bash

#$ -cwd -V
#$-l h_rt=12:59:59
#$ -pe smp 20
#$ -R y
echo "start on `date`, running on `host`"
python ResNet_implementation.py
echo "end on `date`"
