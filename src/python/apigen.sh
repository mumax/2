#! /bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../libmumax2:/usr/local/cuda/lib64
../../bin/apigen $@
