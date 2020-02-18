#!/bin/bash -l
#SBATCH -q debug
#SBATCH -A m3578
#SBATCH -C knl,quad,cache
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J soap
#SBATCH -o my_job.o%j

echo `which python`
date
srun -n 68 --cpu_bind=cores python -u ${soap_dir}/soap_mpi_vec_skopt_dev.py soap.conf > output.out
date
