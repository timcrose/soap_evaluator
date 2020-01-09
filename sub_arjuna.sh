#!/bin/bash                                                                                      
#SBATCH -J soap # Job name
#SBATCH -n 56

#SBATCH -N 1 # Number of nodes                                          
# SBATCH --array=1-16
#SBATCH --mem=0
# SBATCH --mem-per-cpu=2000
# SBATCH -o j_%j_%a.out # File to which STDOUT will be written %j is the job #
#SBATCH -o j_%j.out
#SBATCH -p cpu

export soap_dir=${HOME}/soap_evaluator_git/soap_evaluator
module unload intel
module load gnu
module unload genarris
export PATH=${HOME}/anaconda/anaconda2/bin:$PATH
source activate ${HOME}/anaconda/anaconda2/envs/soap
export LD_LIBRARY_PATH=$HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${HOME}/anaconda/anaconda2/envs/soap/lib/python2.7:${HOME}/anaconda/anaconda2/envs/soap/lib/python2.7/site-packages:${HOME}/glosim_git/glosim_trose:${HOME}

python ${soap_dir}/batch_soap_glosim_trose.py soap.conf > output.out
