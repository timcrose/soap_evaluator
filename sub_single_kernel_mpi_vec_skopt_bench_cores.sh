#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m1980
#SBATCH -C knl,quad,cache
#SBATCH -N 8
#SBATCH -n 512
#SBATCH -t 04:00:00
#SBATCH -J bench
#SBATCH -o my_job.o%j

echo `which python`
srun -n 512 --cpu_bind=cores python -u ${soap_dir}/soap_mpi_vec_skopt.py soap.conf > output.out
