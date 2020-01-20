#!/bin/bash -l
#SBATCH -q debug
#SBATCH -A m1980
#SBATCH -C knl,quad,cache
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -J soap
#SBATCH -o my_job.o%j

echo `which python`
${soap_dir}/prepare_soap_dirs_single_kernel_mpi_vectorized.sh
srun -n 68 --cpu_bind=cores python -u ${soap_dir}/soap_mpi_vec_nocomm.py soap.conf >> output.out
