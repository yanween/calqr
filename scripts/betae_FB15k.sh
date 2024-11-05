#!/bin/bash
#SBATCH -J betae_FB15k-237_PEpoch_50w_TEMP
#SBATCH --output ../logs/FB15k-operation2/betae_nram_domain_only_pretrain_50w_retrain_e-4.out
#SBATCH -p long
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=6

# load the environment
# module purge
# module load apps/anaconda3/2021.05
# source activate kbcqa

export DATA_PATH=../data_operation2/FB15k-q2b
export SAVE_PATH=../logs/FB15k-operation2/betae_nram_domain_only_pretrain_50w_retrain_e-4
export LOG_PATH=../logs/FB15k-operation2/betae_nram_domain_only_pretrain_50w_retrain_e-4.out
export MODEL=temp
export FAITHFUL=no_faithful

export MAX_STEPS=1000000
export VALID_STEPS=5000
export SAVE_STEPS=10000
export ENT_TYPE_NEIGHBOR=32
export REL_TYPE_NEIGHBOR=64
export PRE_TRAIN_MAX_STEP=500000

python -u ../main.py --cuda --do_train --do_valid --do_test --do_pretrain --do_prevalid --do_pretest \
  --data_path $DATA_PATH --save_path $SAVE_PATH -n 128 -b 512 -d 400 -g 60 \
  -plr 0.0001 -lr 0.0001 --max_steps $MAX_STEPS --valid_steps $VALID_STEPS --save_checkpoint_steps $SAVE_STEPS --pre_train_max_steps $PRE_TRAIN_MAX_STEP \
  --cpu_num 6 --geo beta -betam "(1600,2)" --test_batch_size 32 --tasks "1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up" --print_on_screen \
  --faithful $FAITHFUL --model_mode $MODEL --neighbor_ent_type_samples $ENT_TYPE_NEIGHBOR --neighbor_rel_type_samples $REL_TYPE_NEIGHBOR