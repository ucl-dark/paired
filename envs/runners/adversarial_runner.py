import os
from collections import deque, defaultdict

import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd

from util import is_discrete_actions, get_obs_at_index, set_obs_at_index
import re
from scipy import stats as st

import time

import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use("macOSX")


class AdversarialRunner(object):
    """
    Performs rollouts of an adversarial environment, given 
    protagonist (agent), antogonist (adversary_agent), and
    environment adversary (advesary_env)
    """
    def __init__(
        self,
        args,
        agent,
        venv,
        ued_venv=None,
        adversary_agent=None,
        adversary_env=None,
        train=False,
        level_store_max_size=1000,
        device='cpu'):
        """
        venv: Vectorized, adversarial gym env with agent-specific wrappers.
        agent: Protagonist trainer.
        ued_venv: Vectorized, adversarial gym env with adversary-env-specific wrappers.
        adversary_agent: Antogonist trainer.
        adversary_env: Environment adversary trainer.
        """
        self.args = args

        self.venv = venv
        if ued_venv is None:
            self.ued_venv = venv
        else:
            self.ued_venv = ued_venv # Since adv env can have different env wrappers

        self.is_discrete_actions = is_discrete_actions(self.venv)
        self.is_discrete_adversary_env_actions = is_discrete_actions(self.venv, adversary=True)

        self.agents = {
            'agent': agent,
            'adversary_agent': adversary_agent,
            'adversary_env': adversary_env,
        }

        self.agent_rollout_steps = args.num_steps
        self.adversary_env_rollout_steps = self.venv.adversary_observation_space['time_step'].high[0]

        self.is_dr = args.ued_algo == 'domain_randomization'
        self.is_training_env = args.ued_algo in ['minimax', 'paired', 'flexible_paired']
        self.is_paired = args.ued_algo in ['paired', 'flexible_paired']

        # Track running mean and std of env returns for return normalization
        if args.adv_normalize_returns:
            self.env_return_rms = RunningMeanStd(shape=())

        self.device = device

        if train:
            self.train()
        else:
            self.eval()

        self.reset()

    def reset(self):
        self.num_updates = 0
        self.total_episodes_collected = 0

        max_return_queue_size = 10
        self.agent_returns = deque(maxlen=max_return_queue_size)
        self.adversary_agent_returns = deque(maxlen=max_return_queue_size)

    def train(self):
        self.is_training = True
        [agent.train() if agent else agent for _,agent in self.agents.items()]

    def eval(self):
        self.is_training = False
        [agent.eval() if agent else agent for _,agent in self.agents.items()]

    def state_dict(self):
        agent_state_dict = {}
        optimizer_state_dict = {}
        for k, agent in self.agents.items():
            if agent:
                agent_state_dict[k] = agent.algo.actor_critic.state_dict()
                optimizer_state_dict[k] = agent.algo.optimizer.state_dict()

        return {
            'agent_state_dict': agent_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'agent_returns': self.agent_returns,
            'adversary_agent_returns': self.adversary_agent_returns,
            'num_updates': self.num_updates,
            'total_episodes_collected': self.total_episodes_collected,
        }

    def load_state_dict(self, state_dict):
        agent_state_dict = state_dict.get('agent_state_dict')
        for k,state in agent_state_dict.items():
            self.agents[k].algo.actor_critic.load_state_dict(state)

        optimizer_state_dict = state_dict.get('optimizer_state_dict')
        for k,state in optimizer_state_dict.items():
            self.agents[k].algo.optimizer.load_state_dict(state)

        self.agent_returns = state_dict.get('agent_returns')
        self.adversary_agent_returns = state_dict.get('adversary_agent_returns')
        self.num_updates = state_dict.get('num_updates')
        self.total_episodes_collected = state_dict.get('total_episodes_collected')

    def _get_rollout_return_stats(self, rollout_returns):
        # @todo: need to record agent curricula-specific env metrics:
        # - shortest path length
        # - num blocks
        # - passable ratio
        mean_return = torch.zeros(self.args.num_processes, 1)
        max_return = torch.zeros(self.args.num_processes, 1)
        for b, returns in enumerate(rollout_returns):
            if len(returns) > 0:
                mean_return[b] = float(np.mean(returns))
                max_return[b] = float(np.max(returns))

        stats = {
            'mean_return': mean_return,
            'max_return': max_return,
            'returns': rollout_returns 
        }

        return stats

    def _get_env_stats_multigrid(self, agent_info, adversary_agent_info):
        num_blocks = np.mean(self.venv.get_num_blocks())
        passable_ratio = np.mean(self.venv.get_passable())
        shortest_path_lengths = self.venv.get_shortest_path_length()
        shortest_path_length = np.mean(shortest_path_lengths)

        if 'max_returns' in adversary_agent_info:
            solved_idx = \
                (torch.max(agent_info['max_return'], \
                    adversary_agent_info['max_return']) > 0).numpy().squeeze()
        else:
            solved_idx = (agent_info['max_return'] > 0).numpy().squeeze()

        solved_path_lengths = np.array(shortest_path_lengths)[solved_idx]
        solved_path_length = np.mean(solved_path_lengths) if len(solved_path_lengths) > 0 else 0

        stats = {
            'num_blocks': num_blocks,
            'passable_ratio': passable_ratio,
            'shortest_path_length': shortest_path_length,
            'solved_path_length': solved_path_length
        }

        return stats

    def _get_env_stats_minihack(self, agent_info, adversary_agent_info):
        stats = self._get_env_stats_multigrid(agent_info, adversary_agent_info)

        num_monsters = np.mean(self.venv.get_num_monsters())
        num_lava = np.mean(self.venv.get_num_lava())
        num_walls = np.mean(self.venv.get_num_walls())
        num_doors = np.mean(self.venv.get_num_doors())

        stats.update({
            'num_lava': num_lava,
            'num_monsters': num_monsters,
            'num_doors': num_doors,
            'num_walls': num_walls,
        })

        return stats

    def _get_env_stats(self, agent_info, adversary_agent_info):
        env_name = self.args.env_name
        if env_name.startswith('MultiGrid'):
            stats = self._get_env_stats_multigrid(agent_info, adversary_agent_info)
        elif env_name.startswith('MiniHack'):
            stats = self._get_env_stats_minihack(agent_info, adversary_agent_info)
        else:
            raise ValueError(f'Unsupported environment, {self.args.env_name}')
            
        return stats

    def agent_rollout(self, 
                      agent, 
                      num_steps, 
                      update=False, 
                      is_env=False,
                      fixed_seeds=None):
        args = self.args
        if is_env:
            if self.is_dr:
                obs = self.ued_venv.reset_random() # don't need obs here
                return
            else:
                obs = self.ued_venv.reset() # Prepare for constructive rollout
        else:
            obs = self.venv.reset_agent()

        # Initialize first observation
        agent.storage.copy_obs_to_index(obs,0)
        
        mean_return = 0

        rollout_returns = [[] for _ in range(args.num_processes)]
        for step in range(num_steps):
            if args.render:
                self.venv.render_to_screen()
            # Sample actions
            with torch.no_grad():
                obs_id = agent.storage.get_obs(step)
                fwd_start =time.time()
                value, action, action_log_dist, recurrent_hidden_states = agent.act(
                    obs_id, agent.storage.get_recurrent_hidden_state(step), agent.storage.masks[step])

                if self.is_discrete_actions:
                    action_log_prob = action_log_dist.gather(-1, action)
                else:
                    action_log_prob = action_log_dist

            # Observe reward and next obs
            reset_random = self.is_dr
            _action = agent.process_action(action.cpu())
            if is_env:
                obs, reward, done, infos = self.ued_venv.step_adversary(_action)
            else:
                obs, reward, done, infos = self.venv.step_env(_action, reset_random=reset_random)
                if args.clip_reward:
                    reward = torch.clamp(reward, -args.clip_reward, args.clip_reward)

            # Handle early termination due to cliffhanger rollout
            if not is_env and step >= num_steps - 1:
                if agent.storage.use_proper_time_limits:
                    for i, done_ in enumerate(done):
                        if not done_:
                            infos[i]['cliffhanger'] = True
                            infos[i]['truncated'] = True
                            infos[i]['truncated_obs'] = get_obs_at_index(obs, i)

                done = np.ones_like(done, dtype=np.float)

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    rollout_returns[i].append(info['episode']['r'])

                    if not is_env:
                        self.total_episodes_collected += 1

                        # Handle early termination
                        if agent.storage.use_proper_time_limits:
                            if 'truncated_obs' in info.keys():
                                truncated_obs = info['truncated_obs']
                                agent.storage.insert_truncated_obs(truncated_obs, index=i)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'truncated' in info.keys() else [1.0]
                 for info in infos])
            cliffhanger_masks = torch.FloatTensor(
                [[0.0] if 'cliffhanger' in info.keys() else [1.0]
                 for info in infos])

            agent.insert(
                obs, recurrent_hidden_states, 
                action, action_log_prob, action_log_dist, 
                value, reward, masks, bad_masks, 
                cliffhanger_masks=cliffhanger_masks)

        rollout_info = self._get_rollout_return_stats(rollout_returns)

        # Update non-env agent if required
        if not is_env and update: 
            with torch.no_grad():
                obs_id = agent.storage.get_obs(-1)
                next_value = agent.get_value(
                    obs_id, agent.storage.get_recurrent_hidden_state(-1),
                    agent.storage.masks[-1]).detach()

            agent.storage.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda)

            value_loss, action_loss, dist_entropy, info = agent.update()
            # value_loss, action_loss, dist_entropy, info = 0,0,0,{}

            rollout_info.update({
                'value_loss': value_loss,
                'action_loss': action_loss,
                'dist_entropy': dist_entropy,
                'update_info': info,
            })

            # Compute LZ complexity of action trajectories
            if args.log_action_complexity:
                rollout_info.update({'action_complexity': agent.storage.get_action_complexity()})

        return rollout_info

    def _compute_env_return(self, agent_info, adversary_agent_info):
        args = self.args

        if args.ued_algo == 'paired':
            env_return = torch.max(adversary_agent_info['max_return'] - agent_info['mean_return'], \
                torch.zeros_like(agent_info['mean_return']))

        elif args.ued_algo == 'flexible_paired':
            env_return = torch.zeros_like(agent_info['max_return'], dtype=torch.float, device=self.device)
            adversary_agent_max_idx = adversary_agent_info['max_return'] > agent_info['max_return']
            agent_max_idx = ~adversary_agent_max_idx

            env_return[adversary_agent_max_idx] = \
                adversary_agent_info['max_return'][adversary_agent_max_idx]
            env_return[agent_max_idx] = agent_info['max_return'][agent_max_idx]

            env_mean_return = torch.zeros_like(env_return, dtype=torch.float)
            env_mean_return[adversary_agent_max_idx] = \
                agent_info['mean_return'][adversary_agent_max_idx]
            env_mean_return[agent_max_idx] = \
                adversary_agent_info['mean_return'][agent_max_idx]

            env_return = torch.max(env_return - env_mean_return, torch.zeros_like(env_return))

        elif args.ued_algo == 'minimax':
            env_return = -agent_info['max_return']

        else:
            env_return = torch.zeros_like(agent_info['mean_return'])

        if args.adv_normalize_returns:
            self.env_return_rms.update(env_return.flatten().cpu().numpy())
            env_return /= np.sqrt(self.env_return_rms.var + 1e-8)

        if args.adv_clip_reward is not None:
            clip_max_abs = args.adv_clip_reward
            env_return = env_return.clamp(-clip_max_abs, clip_max_abs)
        
        return env_return

    def run(self):
        args = self.args

        adversary_env = self.agents['adversary_env']
        agent = self.agents['agent']
        adversary_agent = self.agents['adversary_agent']

        # Generate a batch of adversarial environments
        env_info = self.agent_rollout(
            agent=adversary_env,
            num_steps=self.adversary_env_rollout_steps,
            update=False,
            is_env=True)

        # Run protagonist agent episodes
        agent_info = self.agent_rollout(
            agent=agent, 
            num_steps=self.agent_rollout_steps,
            update=self.is_training)

        # Run adversary agent episodes
        adversary_agent_info = defaultdict(float)
        if self.is_paired:
            adversary_agent_info = self.agent_rollout(
                agent=adversary_agent, 
                num_steps=self.agent_rollout_steps, 
                update=self.is_training)

        env_return = self._compute_env_return(agent_info, adversary_agent_info)

        adversary_env_info = defaultdict(float)
        if self.is_training and self.is_training_env:
            with torch.no_grad():
                obs_id = adversary_env.storage.get_obs(-1)
                next_value = adversary_env.get_value(
                    obs_id, adversary_env.storage.get_recurrent_hidden_state(-1),
                    adversary_env.storage.masks[-1]).detach()
            adversary_env.storage.replace_final_return(env_return)
            adversary_env.storage.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            env_value_loss, env_action_loss, env_dist_entropy, info = adversary_env.update()
            adversary_env_info.update({
                'action_loss': env_action_loss,
                'value_loss': env_value_loss,
                'dist_entropy': env_dist_entropy,
                'update_info': info
            })

        if self.is_training:
            self.num_updates += 1

        # === LOGGING ===
        stats = self._get_env_stats(agent_info, adversary_agent_info)
        stats.update({
            'mean_env_return': env_return.mean().item(),
            'adversary_env_pg_loss': adversary_env_info['action_loss'],
            'adversary_env_value_loss': adversary_env_info['value_loss'],
            'adversary_env_dist_entropy': adversary_env_info['dist_entropy'],
        })

        [self.agent_returns.append(r) for b in agent_info['returns'] for r in reversed(b)]
        mean_agent_return = 0
        if len(self.agent_returns) > 0:
            mean_agent_return = np.mean(self.agent_returns)

        mean_adversary_agent_return = 0
        if self.is_paired:
            [self.adversary_agent_returns.append(r) for b in adversary_agent_info['returns'] for r in reversed(b)]
            if len(self.adversary_agent_returns) > 0:
                mean_adversary_agent_return = np.mean(self.adversary_agent_returns)

        stats.update({
            'steps': self.num_updates * args.num_processes * args.num_steps,
            'total_episodes': self.total_episodes_collected,

            'mean_agent_return': mean_agent_return,
            'agent_value_loss': agent_info['value_loss'],
            'agent_pg_loss': agent_info['action_loss'],
            'agent_dist_entropy': agent_info['dist_entropy'],

            'mean_adversary_agent_return': mean_adversary_agent_return,
            'adversary_value_loss': adversary_agent_info['value_loss'],
            'adversary_pg_loss': adversary_agent_info['action_loss'],
            'adversary_dist_entropy': adversary_agent_info['dist_entropy'],
        })

        if args.log_grad_norm:
            agent_grad_norm = np.mean(agent_info['update_info']['grad_norms'])
            adversary_grad_norm = 0
            adversary_env_grad_norm = 0
            if self.is_paired:
                adversary_grad_norm = np.mean(adversary_agent_info['update_info']['grad_norms'])
            if self.is_training_env:
                adversary_env_grad_norm = np.mean(adversary_env_info['update_info']['grad_norms'])
            stats.update({
                'agent_grad_norm': agent_grad_norm,
                'adversary_grad_norm': adversary_grad_norm,
                'adversary_env_grad_norm': adversary_env_grad_norm
            })

        if args.log_action_complexity:
            stats.update({
                'agent_action_complexity': agent_info['action_complexity'],
                'adversary_action_complexity': adversary_agent_info['action_complexity']  
            }) 

        return stats