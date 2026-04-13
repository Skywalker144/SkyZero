#!/bin/bash -eu
set -o pipefail
{
# Shuffles selfplay training data from selfplay/ to shuffleddata/current/
# Usage: shuffle.sh BASEDIR TMPDIR NTHREADS BATCHSIZE

if [[ $# -lt 4 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS BATCHSIZE"
    echo "BASEDIR containing selfplay data and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk"
    echo "NTHREADS number of parallel processes for shuffling"
    echo "BATCHSIZE number of samples per batch for training"
    exit 0
fi
BASEDIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift

#------------------------------------------------------------------------------

OUTDIR=$(date "+%Y%m%d-%H%M%S")
OUTDIRTRAIN=$OUTDIR/train
OUTDIRVAL=$OUTDIR/val

mkdir -p "$BASEDIR"/shuffleddata/$OUTDIR
mkdir -p "$TMPDIR"/train
mkdir -p "$TMPDIR"/val

echo "Beginning shuffle at" $(date "+%Y-%m-%d %H:%M:%S")

# Shuffle training data (97% of files by MD5 hash)
(
    time python ./shuffle.py \
         "$BASEDIR"/selfplay/ \
         -expand-window-per-row 0.3 \
         -taper-window-exponent 0.8 \
         -out-dir "$BASEDIR"/shuffleddata/$OUTDIRTRAIN \
         -out-tmp-dir "$TMPDIR"/train \
         -approx-rows-per-out-file 50000 \
         -num-processes "$NTHREADS" \
         -batch-size "$BATCHSIZE" \
         -min-rows 150000 \
         -keep-target-rows 2100000 \
         -only-include-md5-path-prop-lbound 0.00 \
         -only-include-md5-path-prop-ubound 0.97 \
         -output-npz \
         "$@" \
         2>&1 | tee "$BASEDIR"/shuffleddata/$OUTDIR/outtrain.txt &

    wait
)

# Shuffle validation data (3% of files by MD5 hash)
(
    time python ./shuffle.py \
         "$BASEDIR"/selfplay/ \
         -expand-window-per-row 0.3 \
         -taper-window-exponent 0.8 \
         -out-dir "$BASEDIR"/shuffleddata/$OUTDIRVAL \
         -out-tmp-dir "$TMPDIR"/val \
         -approx-rows-per-out-file 50000 \
         -num-processes "$NTHREADS" \
         -batch-size "$BATCHSIZE" \
         -min-rows 150000 \
         -keep-target-rows 51200 \
         -only-include-md5-path-prop-lbound 0.97 \
         -only-include-md5-path-prop-ubound 1.00 \
         -output-npz \
         "$@" \
         2>&1 | tee "$BASEDIR"/shuffleddata/$OUTDIR/outval.txt &

    wait
)

# Update current symlink atomically
sleep 3

rm -rf "$BASEDIR"/shuffleddata/current_tmp
ln -s $OUTDIR "$BASEDIR"/shuffleddata/current_tmp
rm -rf "$BASEDIR"/shuffleddata/current
mv -Tf "$BASEDIR"/shuffleddata/current_tmp "$BASEDIR"/shuffleddata/current

# Cleanup old shuffle dirs (keep last 5, remove those older than 2 hours)
echo "Cleaning up old dirs"
find "$BASEDIR"/shuffleddata/ -mindepth 1 -maxdepth 1 -type d -mmin +120 | sort | head -n -5 | xargs --no-run-if-empty rm -r

echo "Finished shuffle at" $(date "+%Y-%m-%d %H:%M:%S")
echo ""

exit 0
}
