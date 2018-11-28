
function rms(){
  mv --backup=numbered ${@} ${HOME}/trash
}
rms *.csv *.out
rms `find . -name "test_results"`
rms `find . -name "test.xyz*"`
rms `find . -name "*test*.dat"`
rms `find . -name "*rect*k"`
rms `find . -name "log_ref"`
rms `find . -name "*lock*"`
