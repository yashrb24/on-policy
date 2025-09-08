#!/bin/bash

env="PredatorPrey"
scenario="medium"
algo="rmappo"
exp="pp-exploratory-runs"

num_agents=5

# train param
num_env_steps=3000000
episode_length=40

CUDA_VISIBLE_DEVICES=0 python /Users/yashrb/Projects/on-policy/onpolicy/scripts/train/train_predatorprey.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
 --dim 7 --vision 1 \
--n_rollout_threads 15 --ppo_epoch 15 --num_mini_batch 1 \
--save_interval 200 --log_interval 400 \
--use_transformer_base_actor \
--use_wandb False
#--use_wandb False
#--user_name "yashrb" --wandb_name "on-policy"
