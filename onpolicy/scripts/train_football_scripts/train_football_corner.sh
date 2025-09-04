# #!/bin/sh
# # exp param
# env="Football"
# scenario="academy_corner"
# algo="rmappo" # "mappo" "ippo"
# exp="check"
# seed=1

# # football param
# num_agents=10

# # train param
# num_env_steps=50000000
# episode_length=1000

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

# CUDA_VISIBLE_DEVICES=0 python ../train/train_football.py \
# --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
# --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
# --representation "simple115v2" --rewards "scoring" --n_rollout_threads 50 --ppo_epoch 15 --num_mini_batch 2 \
# --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
# --user_name "yashrb" --wandb_name "on-policy" 

#!/bin/sh

CUDA_DEVICE=${1:-1}

# exp param
env="Football"
scenario="academy_corner"
algo="rmappo"
exp="check"
# football param
num_agents=3
# train param
num_env_steps=25000000
episode_length=200
n_rollout_threads=50
ppo_epoch=15
num_mini_batch=2

echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

# Run multiple seeds
for seed in 8 12 18 35 41; do
    echo "Running seed: $seed on GPU: ${CUDA_DEVICE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python ../train/train_football.py \
    --env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed ${seed} \
    --num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
    --representation "simple115v2" --rewards "scoring" --n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} --num_mini_batch ${num_mini_batch} \
    --save_interval 200000 --log_interval 200000 --use_eval --eval_interval 400000 --n_eval_rollout_threads 100 --eval_episodes 100 \
    --user_name "yashrb" --wandb_name "on-policy"
    
    echo "Completed seed: $seed"
done
