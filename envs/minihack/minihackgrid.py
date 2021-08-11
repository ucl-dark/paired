from collections import defaultdict

import numpy as np
import torch

from minihack import LevelGenerator


MONSTER_TYPES = ['jackal', 'sewer rat', 'grid bug', 'kobold zombie', 'newt']


class Grid(object): # it may be worth subclassing LevelGenerator
    """
    Simple class to wrap the MiniHack level generator.
    This reduces the amount of string manipulation we need to do in adversarial.py
    """
    def __init__(self, width, height):

        self.width=width
        self.height=height
        self.level = LevelGenerator(map=None, w=width, h=height, lit=True, flags=("premapped",))

        # separate the map
        self.finalized = False
        self.agent_start_pos = None
        self.goal_pos = None

        # all non-wall/lava objects
        self.objs = defaultdict(list, 
                        {'m': [], 
                         '+': []})

        self.get_grid_obs()

        self.width = width
        self.height = height

    def get_grid_obs(self):
        """
        Converts the des file to a list for the agent
        """
        self.get_map()
        if not self.finalized:
            self.add_objs_to_map()
        self.grid = [''.join(x) for x in self.map]

        return self.grid

    def get_grid_str(self):
        self.finalized = False
        self.get_map()
        self.finalized = True
        return self.map.__str__()

    def get_map(self):
        map = self.level.get_map_array()
        self.map = np.copy(map)
        if not self.finalized:
            self.add_objs_to_map()
        if self.goal_pos and not self.finalized:
            # add these to the map for the adversary
            self.map[self.goal_pos[0]][self.goal_pos[1]] = '>'
        if self.agent_start_pos and not self.finalized:
            self.map[self.agent_start_pos[0]][self.agent_start_pos[1]] = '<'

    def set(self, x, y, character='.'):

        if x > len(self.grid):
            raise ValueError(f'x value {x} is larger than the grid.')
        elif y > len(self.grid):
            raise ValueError(f'y value {y} is larger than the grid.')

        key = self.get(x, y)
        if key in self.objs.keys():
            self.objs[key].remove([x, y])

        if character in ['.', '-', 'L']:
            self.level.add_terrain(coord=(x,y),flag = character)
        elif character in self.objs.keys():
            key = self.get(x, y)
            self.level.add_terrain(coord=(x,y),flag = '.')
            self.objs[character] += [[x,y]]
        elif character in ['>']:
            self.level.add_terrain(coord=(x, y), flag='.')
            self.goal_pos = (x, y)
        elif character in ['<']:
            self.level.add_terrain(coord=(x, y), flag='.')
            self.agent_start_pos = (int(x), int(y))
        else:
            raise ValueError(f'character not supported.')

        # refresh map
        self.get_map()

        # print('set:', character, flush=True)
        # print('map:', self.map, flush=True)

    def remove_obj(self, x, y):
        ## If adding terrain to a loc with a placeholder obj
        ## Need to remove the placeholder from the objs set.
        key = self.get(x, y)
        self.objs[key].remove([x,y])

    def add_objs_to_map(self):
        ## Add monsters etc to the map with a placeholder
        for key, values in self.objs.items():
            if len(values) > 0:
                for value in values:
                    self.map[value[0]][value[1]] = key

    def finalize_agent_goal(self):
        map = self.level.get_map_str()
        self.level = LevelGenerator(map=map, w=self.width, h=self.height, lit=True, flags=("premapped",))
        ## add the monsters and doors
        for obj, locs in self.objs.items():
            if len(locs) == 0:
                pass
            if obj == 'm':
                for idx, loc in enumerate(locs):
                    self.level.add_monster(name=MONSTER_TYPES[idx % (len(MONSTER_TYPES))], place=(loc[1],loc[0]))
            elif obj == '+':
                for loc in locs:
                    self.level.add_door(state="locked",place=(loc[1],loc[0]))

        ## add the staircases
        self.level.add_stair_down((self.goal_pos[1], self.goal_pos[0]))
        self.map[self.goal_pos[0]][self.goal_pos[1]] = '.'
        self.level.add_stair_up((self.agent_start_pos[1], self.agent_start_pos[0]))
        self.map[self.agent_start_pos[0]][self.agent_start_pos[1]] = '.'
        self.finalized = True

    def get(self, x, y):
        self.get_map()
        return self.map[x][y]

    def encode(self):
        # build current map
        self.get_map()

        # use map as obs
        obs = np.copy(self.map)

        # add staircases
        if self.goal_pos and not self.finalized:
            # add these to the map for the adversary
            obs[self.goal_pos[0]][self.goal_pos[1]] = '>'
        if self.agent_start_pos and not self.finalized:
            obs[self.agent_start_pos[0]][self.agent_start_pos[1]] = '<'

        obs = np.array([ord(x) for row in obs for x in row]).reshape([self.width, self.height])
        return obs
