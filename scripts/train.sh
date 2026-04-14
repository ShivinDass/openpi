# DATASET_NAME=val_5
# export DATASET_NAME=$DATASET_NAME
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py gdg_libero_cotrain_pi0_lora \
#                         --exp-name ${DATASET_NAME}-val_5_human_curated_exact_data --overwrite

DATASET_NAME=val_5
export DATASET_NAME=$DATASET_NAME
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py gdg_libero_single_pi0_lora \
                        --exp-name libero90_finetuned --overwrite