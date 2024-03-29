#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m3578
#SBATCH -C knl,quad,cache
#SBATCH -N 1
#SBATCH -n 68
#SBATCH -t 01:00:00
#SBATCH -J bench
#SBATCH -o my_job.o%j

echo `which python`
date
srun -n 68 --cpu_bind=cores python -u ${soap_dir}/soap_mpi_vec_skopt.py soap.conf > output.out
date
