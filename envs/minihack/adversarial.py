
"""An environment which is built by a learning adversary.

Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""

import random

import gym
from gym.utils import seeding
import networkx as nx
from networkx import grid_graph
import numpy as np
import torch

from minihack.navigation import MiniHackNavigation
from . import register
from . import minihackgrid
from nle import nethack
from nle.nethack import Command
from minihack.tiles.rendering import get_des_file_rendering


MOVE_ACTIONS = tuple(nethack.CompassDirection)
NAVIGATE_ACTIONS = tuple(
    list(MOVE_ACTIONS) + [Command.OPEN,Command.KICK]
)

WALL_VALUES = ['-', 'L']

ACTION_KEYS = {
    'walls': '-',
    'none': '.',
    'lava': 'L',
    'goal': '<',
    'agent': '>',
    'monster': 'm',
    'door': '+'
}

def get_action_space(keywords):
    keywords = keywords.split('_')
    action_space = {}
    for idx, key in enumerate(keywords):
        action_space[idx] = ACTION_KEYS[key]
    return action_space


class MiniHackAdversarialEnv(MiniHackNavigation):
  """Grid world where an adversary build the environment the agent plays.

  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.

  The difference here is we are now in MiniHack!

  Walls and Lava are counted as wells.
  Monsters and doors are not.
  """
  def __init__(self, 
              generator_actions='walls',
              actions='move',
              n_clutter=50, 
              size=15,
              agent_view_size=5, 
              max_steps=250,
              goal_noise=0., 
              random_z_dim=50, 
              choose_goal_last=False, 
              seed=0, fixed_environment=False):
    """Initializes environment in which adversary places goal, agent, obstacles.

    Args:
      n_clutter: The maximum number of obstacles the adversary can place.
      size: The number of tiles across one side of the grid; i.e. make a
        size x size grid.
      agent_view_size: The number of tiles in one side of the agent's partially
        observed view of the grid.
      max_steps: The maximum number of steps that can be taken before the
        episode terminates.
      goal_noise: The probability with which the goal will move to a different
        location than the one chosen by the adversary.
      random_z_dim: The environment generates a random vector z to condition the
        adversary. This gives the dimension of that vector.
      choose_goal_last: If True, will place the goal and agent as the last
        actions, rather than the first actions.
    """
    self.seed(seed)
    self.fixed_environment = fixed_environment # Note, fixed seeding is not yet fully supported in MiniHack

    self.agent_start_pos = None
    self.goal_pos = None

    self.done = False

    self.n_clutter = n_clutter

    self.goal_noise = goal_noise
    self.random_z_dim = random_z_dim
    self.choose_goal_last = choose_goal_last
    self.n_agents = 1

    self.max_dim = size ** 2

    # Add two actions for placing the agent and goal.
    self.adversary_max_steps = self.n_clutter + 2

    # generate empty grid
    self.width=size
    self.height=size
    self._gen_grid(self.width, self.height)

    kwargs = {}

    if actions == 'move':
      kwargs["actions"] = MOVE_ACTIONS
    elif actions == 'navigate':
      kwargs["actions"] = NAVIGATE_ACTIONS

    self.obs_keys = ["glyphs","chars","colors","specials","blstats"]

    kwargs['observation_keys'] = self.obs_keys
    kwargs["savedir"] = None
    super().__init__(des_file=self.grid.level.get_des() , **kwargs)

    # Metrics
    self.reset_metrics()

    # Create spaces for adversary agent's specs.
    self.adversary_action_dim = size**2
    self.generator_actions = generator_actions
    self.action_map = get_action_space(self.generator_actions)

    self.adversary_action_space = gym.spaces.MultiDiscrete([len(self.action_map.keys()),self.adversary_action_dim])

    self.adversary_ts_obs_space = gym.spaces.Box(
        low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
    self.adversary_randomz_obs_space = gym.spaces.Box(
        low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)

    self.image_obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.width, self.height),
        dtype='uint8')

    # Adversary observations are dictionaries containing an encoding of the
    # grid, the current time step, and a randomly generated vector used to
    # condition generation (as in a GAN).
    self.adversary_observation_space = gym.spaces.Dict(
        {'image': self.image_obs_space,
         'time_step': self.adversary_ts_obs_space,
         'random_z': self.adversary_randomz_obs_space})

    obs_space = {x:self.observation_space[x] for x in self.obs_keys}
    obs_space['image'] = self.image_obs_space
    self.observation_space = gym.spaces.Dict(obs_space)

    # NetworkX graph used for computing shortest path
    self.graph = grid_graph(dim=[size, size])
    self.wall_locs = []

  def seed(self, seed=None):
      # super().seed(seed=seed) # Not yet fully supported
      self.level_seed = seed
      self.np_random, seed = seeding.np_random(seed)
      return seed

  def set_obs(self, type="reg"):
    if type == "pixels":
        self.obs_keys = ["pixel","glyphs", "chars", "colors", "specials", "blstats"]
    else:
        self.obs_keys = ["glyphs", "chars", "colors", "specials", "blstats"]

  @property
  def processed_action_dim(self):
    return 1

  def _gen_grid(self, width, height):
    """Grid is initially empty, because adversary will create it."""
    # Create an empty grid
    self.grid = minihackgrid.Grid(width=width, height=height)

  def get_goal_x(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[0]

  def get_goal_y(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[1]

  def reset_metrics(self):
    self.distance_to_goal = -1
    self.n_clutter_placed = 0
    self.n_monsters_placed = 0
    self.n_lava_placed = 0
    self.n_walls_placed = 0
    self.n_doors_placed = 0
    self.deliberate_agent_placement = -1
    self.passable = -1
    self.shortest_path_length = (self.width) * (self.height) + 1

  def place_obj(self, obj):
    """
    randomly place an object in an empty loc
    """
    while True:
      x_rand = self.np_random.randint(0,self.width)
      y_rand = self.np_random.randint(0,self.height)
      if self.grid.get(x_rand, y_rand) == '.':
        self.grid.set(x_rand, y_rand, obj)
        return x_rand, y_rand

  def get_agent_goal_loc(self):
    """
    randomly place an object in an empty loc
    """
    while True:
      x_rand = self.np_random.randint(0,self.width)
      y_rand = self.np_random.randint(0,self.height)
      if (x_rand,y_rand) not in [self.goal_pos, self.agent_start_pos]:
        return (x_rand, y_rand)

  def place_obj_deterministic(self, obj):
    """
    Place in the top left corner, and iterate
    """
    for i in range(self.width*self.height):
      x = i % self.width
      y = i // self.width
      current = self.grid.get(x, y)
      if current != '>':
        self.grid.set(x, y, '.')
        return x, y

  def _get_image_obs(self):
    image = self.grid.encode()
    return image

  def reset(self):
    """Fully resets the environment to an empty grid with no agent or goal."""
    """The actual env (MiniHackMaze) won't change until you compile env."""

    self.graph = grid_graph(dim=[self.width, self.height])
    self.wall_locs = []
    self.des_file = None

    self.step_count = 0
    self.adversary_step_count = 0
    self.adversary_max_steps = self.n_clutter + 2

    self.agent_start_dir = self.np_random.randint(0, 4)

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    self.agent_loc = None
    self.goal_loc = None

    self.done = False

    # Extra metrics
    self.reset_metrics()

    # Generate the grid. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self._gen_grid(self.width, self.height)

    super().update(des_file=self.grid.level.get_des())
    env_obs = super().reset()

    obs = {x: env_obs[x] for x in self.obs_keys}

    image = self._get_image_obs()
    obs['image'] = image ## 2d array with string inputs
    obs['time_step'] = [self.adversary_step_count]
    obs['random_z'] = self.generate_random_z()

    return obs

  def reset_agent_status(self):
    """Reset the agent's position, direction, done, and carrying status."""
    self.agent_pos = [None] * self.n_agents
    self.done = [False] * self.n_agents
    self.carrying = [None] * self.n_agents

  def reset_agent(self):
    """Resets the agent's start position, but leaves goal and walls."""

    self.reset_agent_status()
    self.grid.finalized = False
    self.grid.get_map()
    self.grid.finalized = True

    # Step count since episode start
    self.step_count = 0

    # Return first observation
    image = self._get_image_obs()

    env_obs = super().reset()
    obs = {x: env_obs[x] for x in self.obs_keys}
    obs['image'] = image

    return obs

  def reset_to_level(self, level):
    self.reset()
    actions = [np.array(eval(a)) for a in level.split('|')]
    self.adversary_max_steps = len(actions)
    #print("Resetting to level of length: {}".format(len(actions)))
    #print(actions)
    for a in actions:
      obs, _, done, _ = self.step_adversary(a)
      if done:
        self.adversary_max_steps = self.n_clutter + 2
        return self.reset_agent()

  def remove_wall(self, x, y):
    if (x, y) in self.wall_locs:
      self.wall_locs.remove((x, y))
    obj = self.grid.get(x, y)
    if obj != '.':
      self.grid.set(x, y, '.')

  def compute_shortest_path(self):
    if self.agent_start_pos is None or self.goal_pos is None:
      return 0

    self.distance_to_goal = abs(
        self.goal_pos[0] - self.agent_start_pos[0]) + abs(
            self.goal_pos[1] - self.agent_start_pos[1])

    # Check if there is a path between agent start position and goal. Remember
    # to subtract 1 due to outside walls existing in the Grid, but not in the
    # networkx graph.
    self.passable = nx.has_path(
        self.graph,
        source=(self.agent_start_pos[0], self.agent_start_pos[1]),
        target=(self.goal_pos[0], self.goal_pos[1]))
    if self.passable:
      # Compute shortest path
      self.shortest_path_length = nx.shortest_path_length(
          self.graph,
          source=(self.agent_start_pos[0], self.agent_start_pos[1]),
          target=(self.goal_pos[0], self.goal_pos[1]))
    else:
      # Impassable environments have a shortest path length 1 longer than
      # longest possible path
      self.shortest_path_length = (self.width) * (self.height) + 1

  def generate_random_z(self):
    return self.np_random.uniform(size=(self.random_z_dim,)).astype(np.float32)

  def step_adversary(self, action):

    """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.

    The action space is the number of possible squares in the grid. The squares
    are numbered from left to right, top to bottom.

    Args:
      loc: An integer specifying the location to place the next object which
        must be decoded into x, y coordinates.

    Returns:
      Standard RL observation, reward (always 0), done, and info
    """
    if isinstance(self.done, list):
        self.done = self.done[0]
    if self.done:
      image = self.grid.encode()
      obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
      }
      return obs, 0, self.done, {}

    action_dim = 1
    if torch.is_tensor(action) and action.dim() > 0:
        action_dim = 2
    elif len(action) > 1:
        action_dim = 2

    if action_dim == 2:
      obj_idx = action[0]
      if obj_idx in self.action_map.keys():
        obj = self.action_map[obj_idx]
      else:
        obj = '-'
      loc = action[1]
    else:
      obj = '-'
      loc = action.item()

    if loc >= self.adversary_action_dim:
      raise ValueError('Position passed to step_adversary is outside the grid.')

    # Add offset of 1 for outside walls
    x = int(loc % (self.width))
    y = int(loc // (self.width))
    done = False

    should_choose_goal, should_choose_agent = False, False
    if self.choose_goal_last:
        should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
        should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
    else:
        should_choose_goal = self.adversary_step_count == 0
        should_choose_agent = self.adversary_step_count == 1

    # Place goal
    if should_choose_goal:
        self.goal_pos = (x, y)
        self.grid.set(x, y, '>')
    # Place the agent
    elif should_choose_agent:
      if self.grid.get(x, y) == '>':
          self.agent_start_pos = self.place_obj_deterministic('<')
          self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], '.')
          self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], '<')
          self.deliberate_agent_placement = 0
      else:
          self.agent_start_pos = (x, y)
          self.grid.set(x, y, '<')
          self.deliberate_agent_placement = 1
    # Place obj
    elif (self.adversary_step_count < self.adversary_max_steps):
      # If empty, place obj
      if self.grid.get(x, y) == '.':
        self.grid.set(x, y, obj)
      elif self.grid.get(x,y) == '<':
        self.agent_start_pos = None
        self.grid.agent_start_pos = None
        self.grid.set(x, y, obj)
      elif self.grid.get(x,y) == '>':
        self.goal_pos = None
        self.grid.goal_pos = None
        self.grid.set(x, y, obj)
      elif self.grid.get(x, y) != obj:
        self.remove_wall(x,y)
        self.grid.set(x, y, obj)

    self.adversary_step_count += 1

    # End of episode
    if self.adversary_step_count >= self.adversary_max_steps:
      done = True
      self.done = True

    if done:
      if not self.goal_pos:
        self.goal_pos = self.get_agent_goal_loc()
        self.grid.set(self.goal_pos[0], self.goal_pos[1], '>')
      if not self.agent_start_pos:
        self.agent_start_pos = self.get_agent_goal_loc()
        self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], '<')
        self.deliberate_agent_placement = 0

      mapping = [(int(x % (self.width)), int(x // (self.width))) for x in range(self.adversary_action_dim)]
      self.agent_loc = mapping.index(self.agent_start_pos)
      self.goal_loc = mapping.index(self.goal_pos)

      # Reset max steps for level generation
      self.adversary_max_steps = self.n_clutter + 2

      # Finalize level creation
      image = self._get_image_obs()
      self.graph = grid_graph(dim=[self.width, self.height])
      self.wall_locs = [(x, y) for x in range(self.width) for y in range(self.height) if
                        self.grid.get(x, y) in ['-', 'L']]
      for w in self.wall_locs:
        self.graph.remove_node(w)
      try:
        self.compute_shortest_path()
      except:
        print("Steps: {}, AgentPos: {}, GoalPos: {}, NewPoint: ({},{}) {}".format(self.adversary_step_count,
                                                                                  self.agent_start_pos,
                                                                                  self.goal_pos, x,
                                                                                  y, obj), flush=True)
        print(self.grid.map)

      self.grid.finalize_agent_goal()  # Appends the locations to des file
      self.compile_env() # Makes the environment
    else:
      image = self._get_image_obs()

    obs = {
        'image': image,
        'time_step': [self.adversary_step_count],
        'random_z': self.generate_random_z()
    }

    return obs, 0, done, {}

  def step(self, action):
    obs, reward, done, info = super().step(action)

    obs = {k: obs[k] for k in self.obs_keys}

    image = self._get_image_obs()
    obs['image'] = image ## 2D array with string inputs

    return obs, reward, done, info

  def compile_env(self):
      # Updates the des file
      self.des_file = self.grid.level.get_des()
      super().update(self.des_file)

      self.n_monsters_placed = self.des_file.count('MONSTER')
      self.n_doors_placed = self.des_file.count('\nDOOR:')

      map = self.des_file.split('\nMAP\n')[1].split('\nENDMAP\n')[0]
      self.n_lava_placed = map.count('L')
      self.n_walls_placed = map.count('-')

      self.n_clutter_placed = self.n_walls_placed + self.n_lava_placed + self.n_doors_placed + self.n_monsters_placed

  def reset_random(self):
    if self.fixed_environment:
      self.seed(self.level_seed)

    """Use domain randomization to create the environment."""
    self.graph = grid_graph(dim=[self.width, self.height])

    self.step_count = 0
    self.adversary_step_count = 0

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Create empty grid
    self._gen_grid(self.width, self.height)

    # Randomly place goal
    self.goal_pos = self.place_obj('>')

    # Randomly place agent
    #self.agent_start_dir = self._rand_int(0, 4)
    self.agent_start_pos = self.place_obj('<')

    # Randomly place walls
    for _ in range(int(self.n_clutter / 2)):
      wall_loc = self.place_obj('-')
      if (wall_loc[0] - 1, wall_loc[1] - 1) not in self.wall_locs:
        self.wall_locs.append((wall_loc[0] - 1, wall_loc[1] - 1))

    self.compute_shortest_path()
    self.n_clutter_placed = int(self.n_clutter / 2)

    return self.reset_agent()

  def get_grid_str(self):
    return self.grid.get_grid_str()

  def get_des_file(self):
    return self.des_file

  def get_agent_pos_action(self):
    # returns the action to set the agent pos
    if '<' in self.action_map.values():
      return list(self.action_map.values()).index('<')
    else:
      return -1

  def get_goal_pos_action(self):
    # returns the action to set the goal pos
    if '>' in self.action_map.values():
      return list(self.action_map.values()).index('>')
    else:
      return -1

  def get_termination_action(self):
    # returns the action to terminate
    if 'T' in list(self.action_map.values()):
      return list(self.action_map.values()).index('T')
    else:
      return -1

  def render(self, mode='level'):
    des_file = self.des_file
    return np.asarray(get_des_file_rendering(des_file))


class MiniHackGoalLastAdversarialEnv15(MiniHackAdversarialEnv):
  def __init__(self, seed=0, fixed_environment=False):
    super().__init__(generator_actions="walls", n_clutter=85, size=15, choose_goal_last=True)

class MiniHackGoalLastAdversarialEnvWL15(MiniHackAdversarialEnv):
  def __init__(self, seed=0, fixed_environment=False):
    super().__init__(generator_actions="walls_lava", n_clutter=85, size=15,choose_goal_last=True)

class MiniHackGoalLastAdversarialEnvWLMD15(MiniHackAdversarialEnv):
  def __init__(self, seed=0, fixed_environment=False):
    super().__init__(generator_actions="walls_lava_monster_door", n_clutter=85, size=15, choose_goal_last=True)


if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


register.register(
    env_id='MiniHack-GoalLastAdversarial-15x15-v0',
    entry_point=module_path + ':MiniHackGoalLastAdversarialEnv15'
)

register.register(
    env_id='MiniHack-GoalLastAdv-WallsLava-15x15-v0',
    entry_point=module_path + ':MiniHackGoalLastAdversarialEnvWL15'
)

register.register(
    env_id='MiniHack-GoalLastAdv-WallsLavaMonsterDoor-15x15-v0',
    entry_point=module_path + ':MiniHackGoalLastAdversarialEnvWLMD15'
)
