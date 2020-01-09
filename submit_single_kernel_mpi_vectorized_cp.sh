#!/bin/bash                                                                                      
#SBATCH -J soap # Job name
#SBATCH -N 1
#SBATCH -n 56
#SBATCH --mem=0
#SBATCH -o j_%j.out
#SBATCH -p cpu

${soap_dir}/prepare_soap_dirs_single_kernel_mpi_vectorized.sh
mpirun -n 56 python -m cProfile -s cumtime ${soap_dir}/batch_soap_glosim_mpi_vectorized.py soap.conf > output.out
