#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:35:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=toy_example.out

# module load gpu
# module load mamba
# source activate atmt
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# clean up from previous runs
# rm -rf toy_example/data/prepared
# rm -rf toy_example/tokenizers

python train.py \
    --data toy_example/data/prepared/ \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 32 \
    --arch transformer \
    --max-epoch 3 \
    --log-file toy_example/logs/train.log \
    --save-dir toy_example/checkpoints/ \
    --restore-file checkpoint_last.pt \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 100 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

# python translate.py \
#     --input toy_example/data/raw/test.cz \
#     --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
#     --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
#     --checkpoint-path toy_example/checkpoints/checkpoint_best.pt \
#     --batch-size 1 \
#     --max-len 100 \
#     --output toy_example/toy_example_output.en \
#     --bleu \
#     --reference toy_example/data/raw/test.en