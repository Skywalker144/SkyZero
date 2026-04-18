#!/bin/bash -eu
set -o pipefail
{

if [[ $# -lt 2 ]]
then
    echo "Usage: $0 BASEDIR BATCHSIZE [OTHERARGS]"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "BATCHSIZE number of samples per batch for training, must match shuffle"
    exit 0
fi
BASEDIR="$1"
shift
BATCHSIZE="$1"
shift

mkdir -p "$BASEDIR"/train/skyzero
mkdir -p "$BASEDIR"/torchmodels_toexport

time python ./train.py \
     -traindir "$BASEDIR"/train/skyzero \
     -datadir "$BASEDIR"/shuffleddata/current/ \
     -exportdir "$BASEDIR"/torchmodels_toexport \
     -exportprefix skyzero \
     -pos-len 15 \
     -batch-size "$BATCHSIZE" \
     -num-planes 4 \
     -model-config b6c96 \
     -max-epochs-this-instance 1 \
     -samples-per-epoch 2000000 \
     -lr 1e-4 \
     -lr-scale 1.0 \
     -weight-decay 3e-5 \
     -use-fp16 \
     -lookahead-k 6 \
     -lookahead-alpha 0.5 \
     -brenorm-target-rmax 3.0 \
     -brenorm-target-dmax 5.0 \
     -brenorm-adjustment-scale 50000000 \
     "$@" \
     2>&1 | tee -a "$BASEDIR"/train/skyzero/stdout.txt

exit 0
}
