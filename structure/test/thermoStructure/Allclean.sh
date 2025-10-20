#!/bin/sh

domainFluid=${PWD}/dummyFluid
domainStructure=${PWD}/structureDomain

cd ${domainFluid}
cd ..

cd ${domainStructure}
rm -r *.h5
rm -r *.xdmf
rm -r *.png
rm -r *.dat
cd ..

rm output.log

# ----------------------------------------------------------------- end-of-file