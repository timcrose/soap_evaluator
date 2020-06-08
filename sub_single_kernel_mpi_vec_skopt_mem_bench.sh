#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m3578
#SBATCH -C knl,quad,cache
#SBATCH --mem=0
#SBATCH -N 2
#SBATCH -n 136
#SBATCH -t 00:20:00
#SBATCH -J bench
#SBATCH -o my_job.o%j

echo `which python`
date
srun -n 136 --cpu_bind=cores python -u ${soap_dir}/soap_mpi_vec_skopt_mem.py soap.conf > output.out
date
