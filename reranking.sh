cwd=`pwd`
${soap_dir}/clean_reranking.sh
python ${soap_dir}/reranking.py ${cwd}/soap.conf
