#!/bin/bash
# Auto-restart wrapper for the full-scale Stage A run. If training exits for
# ANY reason (OOM, crash, accidental kill, disconnect), automatically relaunches
# it -- resumes cleanly via train.py's own confirmed latest.pth auto-resume logic,
# down to the last mid-epoch checkpoint (--iter-resume saves one every 50
# iterations to iter.pth), not just the last full epoch -- confirmed essential
# during real training, see EXPERIMENT_LOG.md's "serious bug #5" entry.
# Stops automatically once train.py exits with code 0 (genuine, successful completion).

cd /home/user/Documents/min/GaussianFormer3D
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

MAX_RETRIES=50
RETRY_DELAY=60

for i in $(seq 1 $MAX_RETRIES); do
    echo "=== Attempt $i at $(date) ==="
    WANDB_MODE=disabled PYTHONNOUSERSITE=1 python train.py \
        --py-config config/nuscenes_surroundocc_gs25600_full.py \
        --work-dir out/nuscenes_surroundocc_gs25600_full/ \
        --iter-resume
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== Training completed successfully at $(date) ==="
        exit 0
    else
        echo "=== Attempt $i failed (exit code $EXIT_CODE) at $(date) -- retrying in ${RETRY_DELAY}s ==="
        sleep $RETRY_DELAY
    fi
done
echo "=== Gave up after $MAX_RETRIES attempts at $(date) -- needs manual investigation ==="
