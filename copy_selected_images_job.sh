#!/bin/bash

#$ -cwd -V
#$-l h_rt=3:59:59
#$ -pe smp 10
#$ -R y

python ResNet_implementation.py
