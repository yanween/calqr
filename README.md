## Dependencies

- conda create -n temp python=3.7 -y
- PyTorch 1.8.1
- tensorboardX 2.5.1
- numpy 1.21.6

## Running the code

### Dataset

- Download the datasets from [here](https://drive.google.com/drive/folders/1WExXXYDY-66bu0umOd4WAipZUtR36X-M?usp=sharing).
- Create the root directory ./data and put the datasets in.

### Training Model

- Take the GQE model in the FB15k-237 dataset as an example:

  > ```
  > export DATA_PATH=../data_operation2/FB15k-237-q2b
  > export SAVE_PATH=../logs/FB15k-237-operation2/gqe_nram_domain_only_pretrain_retrain
  > export LOG_PATH=../logs/FB15k-237-operation2/gqe_nram_domain_only_pretrain_retrain.out
  > export MODEL=temp
  > export FAITHFUL=no_faithful
  > export NEGATION=baseline
  > export NEGATION_ENHANCE=together
  > export ENHANCE_MODE=domain_only
  > export RANGE_ENHANCE_MODE=None
  > 
  > export MAX_STEPS=450000
  > export VALID_STEPS=5000
  > export SAVE_STEPS=10000
  > export ENT_TYPE_NEIGHBOR=32
  > export REL_TYPE_NEIGHBOR=64
  > export PRE_TRAIN_MAX_STEP=200000
  > 
  > CUDA_VISIBLE_DEVICES=0 nohup python -u ../main.py --cuda --do_train --do_valid --do_test --do_pretrain --do_prevalid --do_pretest \
  >   --data_path $DATA_PATH --save_path $SAVE_PATH -n 128 -b 512 -d 800 -g 24 \
  >   -plr 0.0001 -lr 0.00001 --max_steps $MAX_STEPS --valid_steps $VALID_STEPS --save_checkpoint_steps $SAVE_STEPS --pre_train_max_steps $PRE_TRAIN_MAX_STEP \
  >   --cpu_num 4 --geo vec --test_batch_size 128 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up" --print_on_screen \
  >   --faithful $FAITHFUL --model_mode $MODEL --neighbor_ent_type_samples $ENT_TYPE_NEIGHBOR --neighbor_rel_type_samples $REL_TYPE_NEIGHBOR --negation_mode $NEGATION --negation_together_enhance $NEGATION_ENHANCE --enhance_mode $ENHANCE_MODE --range_enhance_mode $RANGE_ENHANCE_MODE \
  >   > $LOG_PATH 2>&1 &
  > ```

- Other running scripts can be seen in ./scripts.
