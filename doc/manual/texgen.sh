#! /bin/bash


ARGV="$@"

# 1) set up environment
set `uname`
if [ "$1" == "Darwin" ]
then
	INITIALPATH=$PWD
	cd ../../bin/
	MUMAX2BIN=$PWD
	cd $INITIALPATH
else
	MUMAX2BIN=$(dirname $(readlink -f $0)) # path to this script
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUMAX2BIN/../src/libmumax2
export PYTHONPATH=$PYTHONPATH:$MUMAX2BIN/../src/python

exec $MUMAX2BIN/../src/mumax2/texgen $ARGV
