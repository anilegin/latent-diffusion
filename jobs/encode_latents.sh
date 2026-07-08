#!/bin/bash
#SBATCH --job-name=vae-coco256-encode-latents
# Replace with your Slurm account, or remove if unused.
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Path to this repository checkout.
PROJECT_DIR="$HOME/projects/latent-diffusion"
# Path to the Python virtual environment.
VENV_DIR="$PROJECT_DIR/ldm_env"

CONFIG_PATH="configs/experiments/vae_coco_256.yaml"

cd "$PROJECT_DIR"
mkdir -p logs outputs/vae outputs/samples cache

module purge
module load python/3.11.7
module load cuda/12.6

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Useful for CUDA memory fragmentation issues.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

echo "============================================="
echo "Encoding latents for diffusion training..."
echo "============================================="

python scripts/encode_latents.py \
  --config outputs/vae/vae_coco_256_small/config_used.yaml \
  --checkpoint outputs/vae/vae_coco_256_small/checkpoints/best.pt \
  --split train2017 \
  --num-images -1 \
  --batch-size 32 \
  --shard-size 10000 \
  --output-dir outputs/latents/coco_train2017_vae8_scaled_allcaptions \
  --overwrite

echo "============================================="
echo "Train split is done..."
echo "============================================="

python scripts/encode_latents.py \
  --config outputs/vae/vae_coco_256_small/config_used.yaml \
  --checkpoint outputs/vae/vae_coco_256_small/checkpoints/best.pt \
  --split val2017 \
  --num-images -1 \
  --batch-size 32 \
  --shard-size 5000 \
  --output-dir outputs/latents/coco_val2017_vae8_scaled_allcaptions \
  --overwrite

echo "============================================="
echo "Encoding finished: $(date)"
echo "Output directory: outputs/latents/coco_train2017_vae8_scaled_allcaptions"
echo "Output directory: outputs/latents/coco_val2017_vae8_scaled_allcaptions"
echo "============================================="
