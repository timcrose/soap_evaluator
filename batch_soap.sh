rm nohup.out
cwd=`pwd`
for i in {1..12}
do
 nohup python ${soap_dir}/batch_soap.py ${cwd}/soap.conf &
done
