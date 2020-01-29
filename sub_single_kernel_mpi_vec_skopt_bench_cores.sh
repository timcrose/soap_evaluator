#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m1980
#SBATCH -C knl,quad,cache
#SBATCH -N 3
#SBATCH -n 192
#SBATCH -t 03:00:00
#SBATCH -J soap
#SBATCH -o my_job.o%j

echo `which python`
srun -n 192 --cpu_bind=cores python -m cProfile -s cumtime ${soap_dir}/soap_mpi_vec_skopt_cprofile.py soap.conf > output.out
