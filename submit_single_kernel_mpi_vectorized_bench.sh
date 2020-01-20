#!/bin/bash -l
#SBATCH -q debug
#SBATCH -A m1980
#SBATCH -C knl,quad,cache
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -J soapbench
#SBATCH -o my_job.o%j

echo `which python`
${soap_dir}/prepare_soap_dirs_single_kernel_mpi_vectorized.sh
srun -n 257 --cpu_bind=cores python -m cProfile -s cumtime ${soap_dir}/batch_soap_glosim_mpi_vectorized.py soap.conf > output.out
