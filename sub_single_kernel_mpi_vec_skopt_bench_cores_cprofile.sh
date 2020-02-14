#!/bin/bash -l
#SBATCH -q debug
#SBATCH -A m1980
#SBATCH -C knl,quad,cache
#SBATCH -N 1
#SBATCH -n 68
#SBATCH -t 00:10:00
#SBATCH -J cprof
#SBATCH -o my_job.o%j

echo `which python`
hostname
date
srun -n 68 --cpu_bind=cores python -m cProfile -s cumtime ${soap_dir}/soap_mpi_vec_skopt_cprofile.py soap.conf > output.out
date
