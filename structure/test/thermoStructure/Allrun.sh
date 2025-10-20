#!/bin/sh

# Run from this directory
cd ${0%/*} || exit 1

export PYTHONPATH=${PWD}/../../src:$PYTHONPATH
export PYTHONPATH=${PWD}../../../../../coupling_lib/MUI/wrappers/Python:$PYTHONPATH

domainFluid=${PWD}/dummyFluid
domainStructure=${PWD}/structureDomain

# Ranks set to each domain
numProcsFluid=1
numProcsStructure=1

solverFluid=./PUSHER_FETCHER_1
solverStructure=thermalStructure.py

cd ${domainFluid}
cd ..

# parallel run
mpirun -np ${numProcsStructure} -wdir ${domainStructure} python3 -m mpi4py ${solverStructure} 2>&1 | tee output.log

echo "Done"

# ----------------------------------------------------------------- end-of-file