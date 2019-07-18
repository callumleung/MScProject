#!/bin/bash

#$ -cwd -V
#$-l h_rt=12:59:59
#$ -pe smp 10
#$ -R y

python ResNet_implementation.py

