source /opt/conda/etc/profile.d/conda.sh
conda activate ./.conda

python tools/test.py \
    --cfg experiments/deepfashion2/hrnet/w48_384x288_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth \
    TEST.USE_GT_BBOX True
