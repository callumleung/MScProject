!#/bin/bash

#$ -cwd -V
#Â$-l h_rt=3:59:59
#$ -pe smp 10
#$ -R y

python LeNet_implementation.py
