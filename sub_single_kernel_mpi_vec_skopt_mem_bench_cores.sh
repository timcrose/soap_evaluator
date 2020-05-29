#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m3578
#SBATCH -C knl,quad,cache
#SBATCH --mem=0
#SBATCH -N 32
#SBATCH -n 2176
#SBATCH -t 24:00:00
#SBATCH -J bench
#SBATCH -o my_job.o%j

echo `which python`
date
srun -n 2176 --cpu_bind=cores python -u ${soap_dir}/soap_mpi_vec_skopt.py soap.conf > output.out
date
