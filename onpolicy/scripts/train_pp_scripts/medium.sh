#!/bin/bash

env="PredatorPrey"
scenario="medium"
algo="rmappo"
exp="actor-only-transformer-base-ddcl-normalized-loss"

num_agents=5

# train param
num_env_steps=200000
episode_length=40

CUDA_VISIBLE_DEVICES=0 python /Users/yashrb/Projects/on-policy/onpolicy/scripts/train/train_predatorprey.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
 --dim 10 --vision 1 \
--n_rollout_threads 10 --ppo_epoch 10 --num_mini_batch 1 \
--save_interval 200 --log_interval 40 \
--use_transformer_base_actor --n_embd 128 --hidden_size 128 --clip_param 0.2 \
--user_name "yashrb" --wandb_name "on-policy"
#--use_wandb False
#--use_wandb False #--user_name "yashrb" --wandb_name "on-policy"
