#! /bin/bash


MUMAX2BIN=$(dirname $(readlink -f $0))/../../bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUMAX2BIN/../src/libmumax2
export PYTHONPATH=$PYTHONPATH:$MUMAX2BIN/../src/python
exec $MUMAX2BIN/../src/mumax2/texgen $@
