#! /bin/bash

# Produces a tarball with mumax binaries

dir=mumax2
tarball=$dir.tar.gz
rm -rf $dir
mkdir $dir
rm -f $tarball
rm -rf examples/*.out tests/*.out
rm -f test.log
files=$(ls | grep -v pack.sh)
cp -rv $files $dir

echo packing into $tarball: $files

rm -rf $tarball.tar.gz
clean_output="rm -rf examples/*.out tests/*.out src/*.mod/tests/*.out"

make clean && make -j 4 && make test && make -C src/muview && make doc && $clean_output && tar cv $dir | gzip > $tarball

rm -rf $dir

