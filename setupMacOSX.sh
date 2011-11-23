#! /bin/bash

# Run this script once FROM ITS ORIGINAL LOCATION to set up mumax2 for macOSX.
# Not needed on Linux.

if [ ! -e ./mumax2.linux ]; then cp bin/mumax2 ./mumax2.linux; fi
PWD=$(pwd)/bin
sed "s-REPLACE_ME_BY_PATH-$PWD-g" ./mumax2.linux > bin/mumacs2 # todo, replace this by mumax2 for the release. Now it would interfere with git
chmod u+x bin/mumacs2
