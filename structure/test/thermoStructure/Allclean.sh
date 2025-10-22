#!/bin/sh

domainFluid=${PWD}/dummyFluid
domainStructure=${PWD}/structureDomain

cd ${domainFluid}
rm -f *.log
rm heatFluxCpp.txt
rm PUSHER_FETCHER_1
cd build make clean
cd ..
rm -fr build

cd ${domainStructure}
rm -r *.h5
rm -r *.xdmf
rm -r *.png
rm -r *.dat
cd ..

rm output.log

# ----------------------------------------------------------------- end-of-file