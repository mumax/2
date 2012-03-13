#! /bin/bash

# Produces a tarball with mumax binaries

cp -f src/Optimize.inc src/Make.inc

clean_output="rm -rf examples/*.out tests/*.out src/*.mod/tests/*.out"
#make clean && make -j 4 && make test && make -C src/libomf && make -C src/muview && make doc && $clean_output 
make clean && make -j 4 && make -C src/libomf && make -C src/muview && make doc && $clean_output 

if (( $?==0 )); then
	echo build ok;
else
	echo failed;
	exit; 
fi

dir=mumax2
tarball=$dir.tar.gz

rm -rf $dir
mkdir $dir
rm -f $tarball
rm -rf examples/*.out tests/*.out
rm -f test.log
rm -f src/libmumax2/*.o
rm -f src/libomf/*.o
rm -f src/muview/*.o
rm -rf pkg/
files=$(ls | grep -v pack.sh)
cp -rv $files $dir

echo packing into $tarball: $files

rm -rf $tarball.tar.gz

rm -rf $dir/mumax2
tar cv $dir | gzip > $tarball

#rm -rf $dir

cp -f src/Debug.inc src/Make.inc
