#!/bin/bash -eu
set -o pipefail
{
# Takes checkpoints in torchmodels_toexport/ and exports TorchScript .pt files to models/
# Usage: export.sh BASEDIR

if [[ $# -lt 1 ]]
then
    echo "Usage: $0 BASEDIR"
    echo "BASEDIR containing torchmodels_toexport/ and models/ directories"
    exit 0
fi
BASEDIR="$1"
shift

#------------------------------------------------------------------------------

mkdir -p "$BASEDIR"/torchmodels_toexport
mkdir -p "$BASEDIR"/models

# Sort by timestamp so we process oldest to newest
for FILEPATH in $(find "$BASEDIR"/torchmodels_toexport/ -mindepth 1 -maxdepth 1 -type d ! -name "*.tmp" ! -name "*.exported" | sort)
do
    NAME="$(basename "$FILEPATH")"
    echo "Found model to export: $FILEPATH"

    SRC="$BASEDIR/torchmodels_toexport/$NAME"
    TMPDST="$BASEDIR/torchmodels_toexport/$NAME.exported"
    TARGET="$BASEDIR/models/$NAME.pt"

    if [ -f "$TARGET" ]; then
        echo "Model already exists, skipping: $TARGET"
        rm -rf "$SRC"
        continue
    fi

    rm -rf "$TMPDST"
    mkdir -p "$TMPDST"

    set -x
    python ./export_model.py \
        -checkpoint "$SRC/model.ckpt" \
        -output "$TMPDST/model.pt" \
        -board-size 15 \
        -num-planes 4 \
        -model-config b6c96 \
        -use-swa \
        -calibration-data-dir "$BASEDIR/shuffleddata/current/train"
    set +x

    # Create selfplay output directory for this model
    mkdir -p "$BASEDIR/selfplay/$NAME"

    # Move exported model to models/
    mv "$TMPDST/model.pt" "$TARGET"

    # Cleanup
    rm -rf "$SRC"
    rm -rf "$TMPDST"

    echo "Done exporting: $NAME -> $TARGET"
    sleep 2
done

echo "Export complete."
exit 0
}
