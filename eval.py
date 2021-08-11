import sys
import os
import csv
import json
import argparse
import fnmatch
import re
from collections import defaultdict

import numpy as np
import torch
from baselines.common.vec_env import DummyVecEnv
from baselines.logger import HumanOutputFormat
from tqdm import tqdm

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use("macOSX")

import minihack

from envs.registration import make as gym_make
from gym.envs.registration import make as minihack_gym_make
from envs.multigrid.maze import *
from envs.multigrid.crossing import *
from envs.multigrid.fourrooms import *
from envs.wrappers import VecMonitor, VecPreprocessImageWrapper, ParallelAdversarialVecEnv
from util import DotDict, str2bool, make_agent, create_parallel_env
from arguments import parser

"""
Usage:
python -m eval \
--env_name=MultiGrid-SixteenRooms-v0 \
--xpid=<xpid> \
--num_processes=2 \
--base_path="~/logs/paired" \
--result_path="results/"
--verbose
"""

def parse_args():
	parser = argparse.ArgumentParser(description='RL')

	parser.add_argument(
		'--base_path',
		type=str,
		default='~/logs/paired',
		help='Base path to experiment results directories.')
	parser.add_argument(
		'--xpid',
		type=str,
		default='latest',
		help='xpid for evaluation')
	parser.add_argument(
		'--prefix',
		type=str,
		default=None,
		help='xpid prefix for evaluation'
	)
	parser.add_argument(
		'--env_names',
		type=str,
		default='MultiGrid-Labyrinth-v0',
		help='csv of env names')
	parser.add_argument(
		'--result_path',
		type=str,
		default='results/',
		help='Relative path to results directory.')
	parser.add_argument(
		'--accumulator',
		type=str,
		default=None)
	parser.add_argument(
		'--singleton_env',
		type=str2bool, nargs='?', const=True, default=False,
		help="When using a fixed env, whether the same environment should also be reused across workers.")
	parser.add_argument(
		'--seed', 
		type=int, 
		default=1, 
		help='Random seed')
	parser.add_argument(
		'--max_seeds', 
		type=int, 
		default=None, 
		help='Random seed')
	parser.add_argument(
		'--num_processes',
		type=int,
		default=2,
		help='Number of CPU processes to use.')
	parser.add_argument(
		'--max_num_processes',
		type=int,
		default=10,
		help='Number of CPU processes to use.')
	parser.add_argument(
		'--num_episodes',
		type=int,
		default=100,
		help='Number of evaluation episodes.')
	parser.add_argument(
		'--model_tar',
		type=str,
		default='model',
		help='Name of model tar')
	parser.add_argument(
		'--model_name',
		type=str,
		default='agent',
		choices=['agent', 'adversary_agent'],
		help='Which agent to evaluate if more than one option.')
	parser.add_argument(
		'--deterministic',
		type=str2bool, nargs='?', const=True, default=False,
		help="Show logging messages in stdout")
	parser.add_argument(
		'--verbose',
		type=str2bool, nargs='?', const=True, default=False,
		help="Show logging messages in stdout")
	parser.add_argument(
		'--render',
		type=str2bool, nargs='?', const=True, default=False,
		help="Show logging messages in stdout")

	return parser.parse_args()


class Evaluator(object):
	def __init__(self, env_names, num_processes, num_episodes=10, device='cpu', **kwargs):
		self.kwargs = kwargs # kwargs for env wrappers
		self._init_parallel_envs(env_names, num_processes, device=device, **kwargs)
		self.num_episodes = num_episodes

	@property
	def stats_keys(self):
		keys = []
		for env_name in self.env_names:
			keys += [f'solved_rate:{env_name}', f'test_returns:{env_name}']
		return keys

	@staticmethod
	def _get_default_env_kwargs(env_name):
		kwargs = {}
		if env_name.startswith('MiniHack'):
			from envs.minihack.adversarial import NAVIGATE_ACTIONS, MOVE_ACTIONS
			kwargs['actions'] = NAVIGATE_ACTIONS
			kwargs['observation_keys'] = ['glyphs', 'blstats']
			kwargs['savedir'] = None
		
		return kwargs

	@staticmethod
	def make_env(env_name, **kwargs):
		is_multigrid = env_name.startswith('MultiGrid')
		is_minihack = env_name.startswith('MiniHack')

		if is_minihack:
			env_kwargs = Evaluator._get_default_env_kwargs(env_name)
			env = minihack_gym_make(env_name, 
					**Evaluator._get_default_env_kwargs(env_name))
		else:
			env = gym_make(env_name)

		return env

	@staticmethod
	def wrap_venv(venv, env_name, device='cpu'):
		is_multigrid = env_name.startswith('MultiGrid') or env_name.startswith('MiniGrid')
		is_minihack = env_name.startswith('MiniHack')

		transpose_order = [2, 0, 1]
		scale = 10.0
		obs_key = 'image'

		if is_multigrid:
			obs_key = 'image'
			scale = 10.0
		elif is_minihack:
			transpose_order = None
			scale = None
			obs_key = ['glyphs', 'blstats']

		venv = VecMonitor(venv=venv, filename=None, keep_buf=100)	
		venv = VecPreprocessImageWrapper(venv=venv, obs_key=obs_key, 
			transpose_order=transpose_order, scale=scale, device=device)

		return venv

	def _init_parallel_envs(self, env_names, num_processes, device=None, **kwargs):
		self.env_names = env_names
		self.num_processes = num_processes
		self.device = device
		self.venv = {env_name:None for env_name in env_names}

		make_fn = []
		for env_name in env_names:
			make_fn = [lambda: Evaluator.make_env(env_name, **kwargs)]*self.num_processes
			venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
			venv = Evaluator.wrap_venv(venv, env_name, device=device)
			self.venv[env_name] = venv

	def close(self):
		for _, venv in self.venv.items():
			venv.close()

	def evaluate(self, 
		agent, 
		deterministic=False, 
		show_progress=False,
		render=False,
		accumulator='mean'):

		# Evaluate agent for N episodes
		venv = self.venv
		env_returns = {}
		env_solved_episodes = {}
		
		for env_name, venv in self.venv.items():
			returns = []
			solved_episodes = 0

			obs = venv.reset()
			recurrent_hidden_states = torch.zeros(
				self.num_processes, agent.algo.actor_critic.recurrent_hidden_state_size, device=self.device)
			if agent.algo.actor_critic.is_recurrent and agent.algo.actor_critic.rnn.arch == 'lstm':
				recurrent_hidden_states = (recurrent_hidden_states, torch.zeros_like(recurrent_hidden_states))
			masks = torch.ones(self.num_processes, 1, device=self.device)

			pbar = None
			if show_progress:
				print(f'Evaluating on {env_name}')
				pbar = tqdm(total=self.num_episodes)

			while len(returns) < self.num_episodes:
				# Sample actions
				with torch.no_grad():
					_, action, _, recurrent_hidden_states = agent.act(
						obs, recurrent_hidden_states, masks, deterministic=deterministic)

				# Observe reward and next obs
				action = action.cpu()
				action = agent.process_action(action)
				obs, reward, done, infos = venv.step(action)

				masks = torch.tensor(
					[[0.0] if done_ else [1.0] for done_ in done],
					dtype=torch.float32,
					device=self.device)

				for i, info in enumerate(infos):
					if 'episode' in info.keys():
						returns.append(info['episode']['r'])
						if returns[-1] > 0:
							solved_episodes += 1
						if pbar:
							pbar.update(1)

						if len(returns) >= self.num_episodes:
							break

				if render:
					venv.render_to_screen()

			if pbar:
				pbar.close()	
	
			env_returns[env_name] = returns
			env_solved_episodes[env_name] = solved_episodes

		stats = {}
		for env_name in self.env_names:
			if accumulator == 'mean':
				stats[f"solved_rate:{env_name}"] = env_solved_episodes[env_name]/self.num_episodes

			if accumulator == 'mean':
				stats[f"test_returns:{env_name}"] = np.mean(env_returns[env_name])
			else:
				stats[f"test_returns:{env_name}"] = env_returns[env_name]

		return stats


if __name__ == '__main__':
	os.environ["OMP_NUM_THREADS"] = "1"

	display = None
	if sys.platform.startswith('linux'):
		print('Setting up virtual display')

		import pyvirtualdisplay
		display = pyvirtualdisplay.Display(visible=0, size=(1400, 900), color_depth=24)
		display.start()

	args = DotDict(vars(parse_args()))
	args.num_processes = min(args.num_processes, args.num_episodes)

	# === Determine device ====
	device = 'cpu'

	# === Load checkpoint ===
	# Load meta.json into flags object
	base_path = os.path.expandvars(os.path.expanduser(args.base_path))

	xpids = [args.xpid]
	if args.prefix is not None:
		all_xpids = fnmatch.filter(os.listdir(base_path), f"{args.prefix}*")
		filter_re = re.compile('.*_[0-9]*$')
		xpids = [x for x in all_xpids if filter_re.match(x)]

	# Set up results management
	os.makedirs(args.result_path, exist_ok=True)
	if args.prefix is not None:
		result_fname = args.prefix
	else:
		result_fname = args.xpid
	result_fname = f"{result_fname}-{args.model_tar}-{args.model_name}"
	result_fpath = os.path.join(args.result_path, result_fname)
	if os.path.exists(f'{result_fpath}.csv'):
		result_fpath = os.path.join(args.result_path, f'{result_fname}_redo')
	result_fpath = f'{result_fpath}.csv'

	csvout = open(result_fpath, 'w', newline='')
	csvwriter = csv.writer(csvout)

	env_results = defaultdict(list)

	env_names = args.env_names.split(',')

	num_envs = len(env_names)
	if num_envs*args.num_processes > args.max_num_processes:
		chunk_size = args.max_num_processes//args.num_processes
	else:
		chunk_size = num_envs

	num_chunks = int(np.ceil(num_envs/chunk_size))

	num_seeds = 0
	for xpid in xpids:
		if args.max_seeds is not None and num_seeds >= args.max_seeds:
			break

		xpid_dir = os.path.join(base_path, xpid)
		meta_json_path = os.path.join(xpid_dir, 'meta.json')

		model_tar = f'{args.model_tar}.tar'
		checkpoint_path = os.path.join(xpid_dir, model_tar)

		if os.path.exists(checkpoint_path):
			print(f'Evaluating {xpid}')
			meta_json_file = open(meta_json_path)       
			xpid_flags = DotDict(json.load(meta_json_file)['args'])

			make_fn = [lambda: Evaluator.make_env(env_names[0])]
			dummy_venv = ParallelAdversarialVecEnv(make_fn, adversary=False, is_eval=True)
			dummy_venv = Evaluator.wrap_venv(dummy_venv, env_name=env_names[0], device=device)

			# Load the agent
			agent = make_agent(name='agent', env=dummy_venv, args=xpid_flags, device=device)

			try:
				checkpoint = torch.load(checkpoint_path, map_location='cpu')
			except:
				continue
			model_name = args.model_name
			agent.algo.actor_critic.load_state_dict(checkpoint['runner_state_dict']['agent_state_dict'][model_name])

			num_seeds += 1

			# Evaluate environment batch in increments of chunk size
			for i in range(num_chunks):
				start_idx = i*chunk_size
				env_names_ = env_names[start_idx:start_idx+chunk_size]

				# Evaluate the model
				xpid_flags.update(args)
				xpid_flags.update({"use_skip": False})

				evaluator = Evaluator(env_names_, 
					num_processes=args.num_processes, 
					num_episodes=args.num_episodes)

				stats = evaluator.evaluate(agent, 
					deterministic=args.deterministic, 
					show_progress=args.verbose,
					render=args.render,
					accumulator=args.accumulator)

				for k,v in stats.items():
					if args.accumulator:
						env_results[k].append(v)
					else:
						env_results[k] += v

				evaluator.close()
		else:
			print(f'No model path {checkpoint_path}')

	output_results = {}
	for k,_ in stats.items():
		results = env_results[k]
		output_results[k] = f'{np.mean(results):.2f} +/- {np.std(results):.2f}'
		q1 = np.percentile(results, 25, interpolation='midpoint')
		q3 = np.percentile(results, 75, interpolation='midpoint')
		median = np.median(results)
		output_results[f'iq_{k}'] = f'{q1:.2f}--{median:.2f}--{q3:.2f}'
		print(f"{k}: {output_results[k]}")
	HumanOutputFormat(sys.stdout).writekvs(output_results)

	if args.accumulator:
		csvwriter.writerow(['metric',] + [x for x in range(num_seeds)])
	else:
		csvwriter.writerow(['metric',] + [x for x in range(num_seeds*args.num_episodes)])
	for k,v in env_results.items():
		row = [k,] + v
		csvwriter.writerow(row)

	if display:
		display.stop()
