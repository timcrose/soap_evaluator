${soap_dir}/prepare_soap_dirs_single_kernel_mpi_vectorized.sh
mpirun -n 2 python -u ${soap_dir}/batch_soap_glosim_mpi_vectorized.py soap.conf > output.out
