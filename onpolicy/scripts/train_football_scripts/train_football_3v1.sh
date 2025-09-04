# #!/bin/sh
# # exp param
# env="Football"
# scenario="academy_3_vs_1_with_keeper"
# algo="rmappo" # "mappo" "ippo"
# exp="actor-only-transformer-base-ddcl-normalized-loss"

# # football param
# num_agents=3

# # train param
# num_env_steps=25000000
# episode_length=200

# CUDA_VISIBLE_DEVICES=0 python ../train/train_football.py \
# --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
# --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
# --representation "simple115v2" --rewards "scoring" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
# --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
# --use_transformer_base_actor --user_name "yashrb" --wandb_name "on-policy" 
# # --use_comms_channel --comm_coeff 0.0001


#!/bin/bash

CUDA_DEVICE=${1:-0}
SEEDS=(1 2 3 4 5)  # 5 different seeds

env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="rmappo"
exp="actor-only-transformer-base-ddcl-normalized-loss"
num_agents=3
num_env_steps=25000000
episode_length=200

for seed in "${SEEDS[@]}"; do
    echo "Running seed: $seed"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python ../train/train_football.py \
    --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
    --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
    --representation "simple115v2" --rewards "scoring" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
    --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
    --use_transformer_base_actor --user_name "yashrb" --wandb_name "on-policy"
    
    echo "Completed seed: $seed"
done
