#!/bin/bash

#$ -cwd -V
#$ -l h_rt=100:59:59
#$ -l h_vmem=64G
#$ -q gpu.q

echo "start on `date`"
python ResNet_implementation.py
echo "end on `date`"

