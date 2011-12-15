#! /bin/bash

# Produces a tarball with mumax binaries

tarball=mumax2.tar.gz
rm -f $tarball
rm -rf examples/*.out tests/*.out
files=$(ls | grep -v pack.sh)

echo packing into $tarball: $files

rm -rf $tarball.tar.gz
clean_output="rm -rf examples/*.out tests/*.out src/*.mod/tests/*.out"

make clean && make -j 4 && make test && make doc && $clean_output && tar cv $files | gzip > $tarball


