#!/bin/bash                                                                                      
#SBATCH -J soap # Job name
#SBATCH -n 1

#SBATCH -N 1 # Number of nodes                                          
#SBATCH --array=1-16
#SBATCH --mem-per-cpu=1050
#SBATCH -o j_%j_%a.out # File to which STDOUT will be written %j is the job #                                       
#SBATCH -p idle

module unload genarris
source /home/trose/anaconda/anaconda2/bin/deactivate
source /home/trose/anaconda/anaconda2/bin/activate /home/trose/anaconda/anaconda2/envs/soap
export LD_LIBRARY_PATH=$HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/trose/anaconda/anaconda2/envs/soap/lib/python2.7:/home/trose/epfl/epfl_arjuna/epfl/glosim/quippy:$HOME/lib64:$PYTHONPATH
export soap_dir=${HOME}/soap_evaluator_git/soap_evaluator
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ohpc/pub/intel/intel18/compilers_and_libraries_2018.0.128/linux/mkl/lib/intel64_lin #:/home/trose/epfl/epfl/glosim/quippy:$LD_LIBRARY_PATH
#export PYTHONPATH=$PYTHONPATH:$LD_LIBRARY_PATH:/home/trose/anaconda/anaconda2/envs/soap/lib/python2.7:/home/trose/epfl/epfl/glosim/quippy:/home/trose/anaconda/anaconda2/envs/soap/lib/python2.7/site-packages
#echo "python_interp"
#echo `which python`

python ${soap_dir}/batch_soap.py soap.conf > output.out
