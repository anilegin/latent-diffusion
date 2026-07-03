#!/bin/bash
#SBATCH --job-name=ldm-strong-multigpu
#SBATCH --account=iscrc_mnlp26
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --mem=240GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="$HOME/projects/latent-diffusion"
VENV_DIR="$PROJECT_DIR/ldm_env"
CONFIG_PATH="configs/experiments/ldm_coco_256_vpred_strong_ddp.yaml"

cd "$PROJECT_DIR"
mkdir -p logs outputs/ldm

module purge
module load python/3.11.7
module load cuda/12.6

source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "============================================="
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Config: $CONFIG_PATH"
echo "Python: $(which python)"
python --version
echo "============================================="

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=3 \
  scripts/train_ldm_multigpu.py \
  --config "$CONFIG_PATH"

# torchrun \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=2 \
#   scripts/train_ldm_ddp.py \
#   --config outputs/ldm/ldm_coco_256_vae8_strong_vpred_ddp/config_used.yaml \
#   --resume-from outputs/ldm/ldm_coco_256_vae8_strong_vpred_ddp/checkpoints/last.pt