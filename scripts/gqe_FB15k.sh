export DATA_PATH=../data_operation2/FB15k-237-q2b
export SAVE_PATH=../logs/FB15k-operation2/gqe_nram_domain_only_pretrain_retrain_rel_as_type
export LOG_PATH=../logs/FB15k-operation2/gqe_nram_domain_only_pretrain_retrain_rel_as_type.out
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
export PRE_TRAIN_MAX_STEP=1000000

CUDA_VISIBLE_DEVICES=0 nohup python -u ../main.py --cuda --do_train --do_valid --do_test --do_pretrain --do_prevalid --do_pretest \
  --data_path $DATA_PATH --save_path $SAVE_PATH -n 128 -b 512 -d 800 -g 24 \
  -plr 0.0001 -lr 0.00001 --max_steps $MAX_STEPS --valid_steps $VALID_STEPS --save_checkpoint_steps $SAVE_STEPS --pre_train_max_steps $PRE_TRAIN_MAX_STEP \
  --cpu_num 4 --geo vec --test_batch_size 128 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen \
  --faithful $FAITHFUL --model_mode $MODEL --neighbor_ent_type_samples $ENT_TYPE_NEIGHBOR --neighbor_rel_type_samples $REL_TYPE_NEIGHBOR --negation_mode $NEGATION --negation_together_enhance $NEGATION_ENHANCE --enhance_mode $ENHANCE_MODE --range_enhance_mode $RANGE_ENHANCE_MODE \
  > $LOG_PATH 2>&1 &