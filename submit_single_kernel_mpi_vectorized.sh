#!/bin/bash                                                                                      
#SBATCH -J soap # Job name
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem-per-cpu=10000
 #SBATCH --mem=0
#SBATCH -o j_%j.out
#SBATCH -p cpu
# SBATCH --nodelist=d002

echo PATH
echo $PATH
echo " "
echo PYTHONPATH
echo $PYTHONPATH
echo " "
echo LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

${soap_dir}/prepare_soap_dirs_single_kernel_mpi_vectorized.sh
mpirun -n 2 python ${soap_dir}/batch_soap_glosim_mpi_vectorized.py soap.conf > output.out
