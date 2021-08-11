import argparse

import torch

from util import str2bool

parser = argparse.ArgumentParser(description='UED')

# PPO Arguments. 
parser.add_argument(
    '--algo',
    type=str,
    default='ppo',
    choices=['ppo', 'a2c', 'acktr', 'ucb', 'mixreg'],
    help='Which RL algorithm to use')
parser.add_argument(
    '--lr', 
    type=float, 
    default=1e-4, 
    help='PPO learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.995,
    help='Discount factor for rewards')
parser.add_argument(
    '--use_gae',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use generalized advantage estimator.')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='GAE lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.0,
    help='Entropy term coefficient for student agents')
parser.add_argument(
    '--adv_entropy_coef',
    type=float,
    default=0.005,
    help='Entropy term coefficient for adversary')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='Value loss coefficient')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='Max norm of gradients for student agents')
parser.add_argument(
    '--adv_max_grad_norm',
    type=float,
    default=0.5,
    help='Max norm of gradients for adversary')
parser.add_argument(
    '--normalize_returns',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize student agent returns')
parser.add_argument(
    '--adv_normalize_returns',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize adversary returns')
parser.add_argument(
    '--use_popart',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize student values via PopArt')
parser.add_argument(
    '--adv_use_popart',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to normalize adversary values using PopArt')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='Random seed for run')
parser.add_argument(
    '--num_processes',
    type=int,
    default=4,
    help='How many training CPU processes to use for rollouts')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='Number of rollout steps for PPO')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=5,
    help='Number of PPO epochs used by student agents')
parser.add_argument(
    '--adv_ppo_epoch',
    type=int,
    default=5,
    help='Number of PPO epochs used by adversary')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=1,
    help='Number of batches for PPO used by student agents')
parser.add_argument(
    '--adv_num_mini_batch',
    type=int,
    default=1,
    help='number of batches for PPO used by adversary')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='PPO clip parameter')
parser.add_argument(
    '--clip_value_loss',
    type=str2bool,
    default=True,
    help='PPO clip value loss')
parser.add_argument(
    '--adv_clip_reward',
    type=float,
    default=None,
    help='Clip adversary rewards')
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=500000,
    help='Total number of environment steps for training')

# Architecture arguments.
parser.add_argument(
    '--recurrent_arch',
    type=str,
    default='lstm',
    choices=['gru', 'lstm'],
    help='RNN architecture')
parser.add_argument(
    '--recurrent_agent',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use RNN for student agents')
parser.add_argument(
    '--recurrent_adversary_env',
    type=str2bool, nargs='?', const=True, default=False,
    help='Use RNN for adversary')
parser.add_argument(
    '--recurrent_hidden_size',
    type=int,
    default=256,
    help='Recurrent hidden state size')

# Environment arguments.
parser.add_argument(
    '--env_name',
    type=str,
    default='MiniHack-GoalLastAdversarial-v0',
    help='Environment to train on')
parser.add_argument(
    '--handle_timelimits',
    type=str2bool, nargs='?', const=True, default=False,
    help="Bootstrap off of early termination states. Requires env to be wrapped by envs.wrappers.TimeLimit.")
parser.add_argument(
    '--singleton_env',
    type=str2bool, nargs='?', const=True, default=False,
    help="When using a fixed env, whether the same environment should also be reused across workers.")
parser.add_argument(
    '--clip_reward',
    type=float,
    default=None,
    help="Clip sparse rewards")

# UED arguments.
parser.add_argument(
    '--ued_algo',
    type=str,
    default='paired',
    choices=['domain_randomization', 'minimax', 'paired', 'flexible_paired'],
    help='UED algorithm')

# Hardware arguments.
parser.add_argument(
    '--no_cuda',
    type=str2bool, nargs='?', const=True, default=False,
    help='disables CUDA training')

# Logging arguments.
parser.add_argument(
    "--verbose", 
    type=str2bool, nargs='?', const=True, default=True,
    help="Whether to print logs")
parser.add_argument(
    '--xpid',
    default='latest',
    help='Name for the run, also name of directory containing log files')
parser.add_argument(
    '--log_dir',
    default='~/logs/paired/',
    help='Directory in which to log experiments')
parser.add_argument(
    '--log_interval',
    type=int,
    default=1,
    help='Log interval, one log per n updates')
parser.add_argument(
    "--save_interval", 
    type=int, 
    default=20,
    help="Save model every n updates.")
parser.add_argument(
    "--archive_interval", 
    type=int, 
    default=0,
    help="Save an archived model every n updates.")
parser.add_argument(
    "--screenshot_interval", 
    type=int, 
    default=1,
    help="Save screenshot of environment every n updates.")
parser.add_argument(
    '--render',
    type=str2bool, nargs='?', const=True, default=False,
    help='Render one parallel environment to screen during training')
parser.add_argument(
    "--checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Begin training from checkpoint.")
parser.add_argument(
    "--disable_checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Disable saving checkpoint.")
parser.add_argument(
    '--log_grad_norm',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log gradient norms")
parser.add_argument(
    '--log_action_complexity',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log action trajectory complexity metric")
parser.add_argument(
    '--test_interval',
    type=int,
    default=250,
    help='Evaluate on test envs every n updates.')
parser.add_argument(
    '--test_num_episodes',
    type=int,
    default=10,
    help='Number of test episodes per test environment')
parser.add_argument(
    '--test_num_processes',
    type=int,
    default=4,
    help='Number of test CPU processes per test environment')
parser.add_argument(
    '--test_env_names',
    type=str,
    default='MiniHack-Room-15x15-v0,MiniHack-MazeWalk-9x9-v0,MiniHack-MazeWalk-15x15-v0',
    help='Environments to evaluate on (csv string)')
