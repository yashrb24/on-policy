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
--rewards "scoring" --n_rollout_threads 10 --ppo_epoch 10 --num_mini_batch 1 \
--save_interval 200 --log_interval 200 --use_eval --eval_interval 400 --n_eval_rollout_threads 32 --eval_episodes 32 \
--use_transformer_base_actor --use_comms_channel --comm_coeff 0.0001 --use_wandb False
