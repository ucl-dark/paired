# for laptop
# !/bin/bash
python -m train \
  --xpid=test_ued \
  --env_name=MultiGrid-GoalLastAdversarial-v0 \
  --use_gae=True \
  --gamma=0.995 \
  --seed=88 \
  --recurrent_arch=lstm \
  --recurrent_agent=True \
  --recurrent_adversary_env=True \
  --recurrent_hidden_size=256 \
  --lr=0.0001 \
  --num_steps=256 \
  --num_processes=2 \
  --num_env_steps=1000000 \
  --ppo_epoch=5 \
  --num_mini_batch=1 \
  --entropy_coef=0.0 \
  --adv_entropy_coef=0.005 \
  --algo=ppo \
  --ued_algo=paired \
  --test_env_names='MultiGrid-SixteenRooms-v0' \
  --log_interval=1 \
  --screenshot_interval=1 \
  --log_grad_norm=False \
  --log_dir=~/logs/minihack \
  --checkpoint=False \
  --verbose
