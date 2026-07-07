#!/bin/bash
#SBATCH --job-name=vae-coco256-encode-latents-sd15
#SBATCH --account=iscrc_mnlp26
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

PROJECT_DIR="$HOME/projects/latent-diffusion"
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

python scripts/encode_latents_sd15.py \
  --model-id Lykon/dreamshaper-8 \
  --images-dir /leonardo_scratch/large/userexternal/aegin000/datasets/coco2017/images/train2017 \
  --captions-json /leonardo_scratch/large/userexternal/aegin000/datasets/coco2017/annotations/captions_train2017.json \
  --output-dir outputs/latents/coco_train2017_sd15vae_scaled018215_allcaptions \
  --batch-size 32 \
  --shard-size 5000

echo "============================================="
echo "Train split is done..."
echo "============================================="

python scripts/encode_latents_sd15.py \
  --model-id Lykon/dreamshaper-8 \
  --images-dir /leonardo_scratch/large/userexternal/aegin000/datasets/coco2017/images/val2017 \
  --captions-json /leonardo_scratch/large/userexternal/aegin000/datasets/coco2017/annotations/captions_val2017.json \
  --output-dir outputs/latents/coco_val2017_sd15vae_scaled018215_allcaptions \
  --batch-size 32 \
  --shard-size 5000

echo "============================================="
echo "Encoding finished: $(date)"
echo "Output directory: outputs/latents/coco_train2017_sd15vae_scaled018215_allcaptions"
echo "Output directory: outputs/latents/coco_val2017_sd15vae_scaled018215_allcaptions"
echo "============================================="
