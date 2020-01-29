#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m1980
#SBATCH -C knl,quad,cache
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J targ1
#SBATCH -o my_job.o%j

echo `which python`
srun -n 68 --cpu_bind=cores python -u ${soap_dir}/soap_mpi_vec_skopt.py soap.conf > output.out
