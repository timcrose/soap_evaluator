#!/bin/bash -l
#SBATCH -q regular
#SBATCH -A m1980
#SBATCH -C knl,quad,cache
#SBATCH -N 48
#SBATCH -t 10:00:00
#SBATCH -J soap
#SBATCH -o my_job.o%j

echo `which python`
${soap_dir}/prepare_soap_dirs_single_kernel_mpi_vectorized.sh
srun -n 3264  --cpu_bind=cores python -u ${soap_dir}/batch_soap_glosim_mpi_vectorized.py soap.conf >> output.out
