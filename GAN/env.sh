
#! /bin/bash
THISDIR=`pwd`
cd /afs/cern.ch/work/g/gvonsem/public/HGCAL/ML/DeepJetCore
source env.sh
cd $THISDIR
export GAN=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$GAN

cd $GAN
export PYTHONPATH=$GAN/modules:$PYTHONPATH
export PYTHONPATH=$GAN/modules/datastructures:$PYTHONPATH
export PATH=$GAN/scripts:$PATH
