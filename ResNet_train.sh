#!/bin/bash

#$ -cwd -V
#$-l h_rt=12:59:59
#$ -pe mpi 40
#$ -R y

mpirun -np 40 ResNet_implementation.py

