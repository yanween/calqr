#!/bin/bash
#SBATCH -J logice_NELL
#SBATCH --output ../logs/NELL-q2b-operation/logice_nram_domain_only_pretrain_retrain.out
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --ntasks-per-node=6
#SBATCH --time=72:00:00
#SBATCH --mem=60G

# load the environment
# module purge
# module load apps/anaconda3/2021.05
# source activate kbcqa

export DATA_PATH=../data_operation2/NELL-q2b
export SAVE_PATH=../logs/NELL-q2b-operation2/logice_nram_domain_only_pretrain_retrain
export LOG_PATH=../logs/NELL-q2b-operation2/logice_nram_domain_only_pretrain_retrain.out
export MODEL=temp
export FAITHFUL=no_faithful
export NEGATION=baseline
export NEGATION_ENHANCE=together
export ENHANCE_MODE=domain_only
export RANGE_ENHANCE_MODE=None

#export MAX_STEPS=450000
export MAX_STEPS=1000000
export VALID_STEPS=5000
export SAVE_STEPS=10000
export ENT_TYPE_NEIGHBOR=32
export REL_TYPE_NEIGHBOR=64
export PRE_TRAIN_MAX_STEP=200000

python -u ../main.py --cuda --do_train --do_valid --do_test --do_pretrain --do_prevalid --do_pretest \
  --data_path $DATA_PATH --save_path $SAVE_PATH -n 128 -b 512 -d 400 -g 0.375 \
  -plr 0.0001 -lr 0.0001 --max_steps $MAX_STEPS --valid_steps $VALID_STEPS --save_checkpoint_steps $SAVE_STEPS --pre_train_max_steps $PRE_TRAIN_MAX_STEP \
  --cpu_num 6 --geo logic -logicm "(luk,0,1,0,1600,2)" --test_batch_size 16 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen \
  --faithful $FAITHFUL --model_mode $MODEL --neighbor_ent_type_samples $ENT_TYPE_NEIGHBOR --neighbor_rel_type_samples $REL_TYPE_NEIGHBOR --negation_mode $NEGATION --negation_together_enhance $NEGATION_ENHANCE --enhance_mode $ENHANCE_MODE --range_enhance_mode $RANGE_ENHANCE_MODE