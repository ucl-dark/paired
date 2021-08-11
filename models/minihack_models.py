import torch
from torch import nn
from torch.nn import functional as F
import collections

from gym.spaces import MultiDiscrete
from nle import nethack

from typing import NamedTuple, Union
from collections import namedtuple
from .minihack_util import id_pairs_table
import logging

from .common import *
from .distributions import Categorical

## this is a custom network for the adversary, it is like the multigrid network as it uses the map
class MiniHackAdversaryNetwork(DeviceAwareModule):
    def __init__(
        self,
        observation_space,
        action_space,
        obs_key,
        actor_fc_layers=(32, 32),
        value_fc_layers=(32, 32),
        glyph_embedding_dim=32,
        scalar_dim=4,
        scalar_fc=5,
        random_z_dim=0,
        recurrent_arch='lstm',
        recurrent_hidden_size=256,
        use_crop=False,
        random=False
    ):
        super(MiniHackAdversaryNetwork, self).__init__()

        self.obs_key = obs_key

        self.input_shape = observation_space[obs_key].shape
        self.action_space = action_space

        if isinstance(action_space, MultiDiscrete):
            self.num_actions = list(action_space.nvec)
            self.multi_dim = True
            self.action_dim = len(self.num_actions)
            self.num_action_logits = np.sum(list(self.num_actions))
        else:
            self.num_actions = action_space.n
            self.multi_dim = False
            self.action_dim = 1
            self.num_action_logits = self.num_actions

        self.recurrent_arch = recurrent_arch
        self.random = random

        self.H = self.input_shape[0]
        self.W = self.input_shape[1]

        self.k_dim = glyph_embedding_dim
        self.h_dim = recurrent_hidden_size

        conv_kernel_size = 3
        self.image_embedding = nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size, stride=1, padding=0),
            nn.Flatten(),
            nn.ReLU()
        )

        self.image_embedding_size = (self.input_shape[0]-conv_kernel_size+1)**2

        self.scalar_dim = scalar_dim
        if scalar_dim:
            self.use_scalar = True
            self.scalar_embed = nn.Linear(scalar_dim, scalar_fc)
            self.image_embedding_size += scalar_fc
        else:
            self.use_scalar = False

        self.image_embedding_size += random_z_dim

        self.rnn = None
        if recurrent_arch:
            self.rnn = RNN(
                input_size=self.image_embedding_size,
                hidden_size=self.h_dim,
                arch=recurrent_arch)
            self.base_output_size = self.h_dim
            self.recurrent_hidden_state_size = self.h_dim
        else:
            self.base_output_size = self.image_embedding_size
            self.recurrent_hidden_state_size = 0

        if self.multi_dim:
            self.actor_obj = nn.Sequential(
                make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
                Categorical(actor_fc_layers[-1], self.num_actions[0])
            )
            self.actor_loc = nn.Sequential(
                make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
                Categorical(actor_fc_layers[-1], self.num_actions[1])
            )
        else:
            self.actor = nn.Sequential(
                make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
                Categorical(actor_fc_layers[-1], self.num_actions)
            )
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.base_output_size),
            init_(nn.Linear(value_fc_layers[-1], 1))
        )

    @property
    def is_recurrent(self):
        return self.rnn is not None

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def _forward_base(self, inputs, rnn_hxs, masks):

        image = inputs.get(self.obs_key)

        if self.use_scalar:
            scalar = inputs.get('time_step')
            in_scalar = one_hot(self.scalar_dim, scalar).to(self.device)
            in_scalar = self.scalar_embed(in_scalar)
        else:
            in_scalar = torch.tensor([], device=self.device)

        in_z = inputs.get('random_z', torch.tensor([], device=self.device))

        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        image_emb = self.image_embedding(image)
        in_features = torch.cat((image_emb, in_scalar, in_z), dim=-1)

        if self.recurrent_arch:
            core_features, rnn_hxs = self.rnn(in_features, rnn_hxs, masks)
        else:
            core_features = in_features

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):

        if self.random:
            B = inputs['image'].shape[0]
            if self.multi_dim:
                action = torch.zeros((B, 2), dtype=torch.int64, device=self.device)
                values = torch.zeros((B, 1), device=self.device)
                action_log_dist = torch.ones(B, self.action_space.nvec[0] + self.action_space.nvec[1], device=self.device)
                for b in range(B):
                    action[b] = torch.tensor(self.action_space.sample()).to(self.device)
            else:
                action = torch.zeros((B,1), dtype=torch.int64, device=self.device)
                values = torch.zeros((B,1), device=self.device)
                action_log_dist = torch.ones(B, self.action_space.n, device=self.device)
                for b in range(B):
                   action[b] = self.action_space.sample()

            return values, action, action_log_dist, rnn_hxs

        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        if self.multi_dim:
            dist_obj = self.actor_obj(core_features)
            dist_loc = self.actor_loc(core_features)
            action_obj = dist_obj.sample()
            action_loc = dist_loc.sample()
            action = torch.cat((action_obj, action_loc),dim=1)
            action_log_dist = torch.cat((dist_obj.logits, dist_loc.logits),dim=1)
            obj_entropy = dist_obj.entropy().mean()
            loc_entropy = dist_loc.entropy().mean()
        else:
            dist = self.actor(core_features)
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            action_log_dist = dist.logits
            dist_entropy = dist.entropy().mean()

        value = self.critic(core_features)

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        if self.multi_dim:
            dist_obj = self.actor_obj(core_features)
            dist_loc = self.actor_loc(core_features)

            action_obj_log_probs = dist_obj.log_probs(action[:, 0])
            action_loc_log_probs = dist_loc.log_probs(action[:, 1])

            action_log_probs = torch.cat((action_obj_log_probs, action_loc_log_probs),dim=1)

            obj_entropy = dist_obj.entropy().mean()
            loc_entropy = dist_loc.entropy().mean()
            dist_entropy = [obj_entropy,loc_entropy]
        else:
            dist = self.actor(core_features)
            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()

        value = self.critic(core_features)
        return value, action_log_probs, dist_entropy, rnn_hxs


### === Below is taken from the NLE repo === 
def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)

class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )

class NetHackAgentNet(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_actions,
        use_lstm=True,
        rnn_hidden_size=256,
        embedding_dim=32,
        crop_dim=9,
        num_layers=5,
    ):
        super(NetHackAgentNet, self).__init__()

        self.glyph_shape = observation_shape["glyphs"].shape
        self.blstats_size = observation_shape["blstats"].shape[0]

        self.recurrent_hidden_state_size = rnn_hidden_size

        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = 512

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim ** 2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, rnn_hidden_size),
            nn.ReLU(),
        )

        if self.use_lstm:
            self.rnn = RNN(
                input_size=rnn_hidden_size,
                hidden_size=rnn_hidden_size,
                arch='lstm')

        self.policy = Categorical(rnn_hidden_size, self.num_actions)

        self.baseline = nn.Linear(rnn_hidden_size, 1)

    @property
    def is_recurrent(self):
        return self.use_lstm

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def _forward_base(self, env_outputs, core_state, masks, deterministic=False):
        # -- [T x B x H x W]
        glyphs = env_outputs["glyphs"]
        glyphs = glyphs.unsqueeze(0)
        # -- [T x B x F]
        blstats = env_outputs["blstats"]

        T, B, *_ = glyphs.shape

        # -- [B' x H x W]
        glyphs = torch.flatten(glyphs, 0, 1)  # Merge time and batch.

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]
        # TODO ???
        # coordinates[:, 0].add_(-1)

        # -- [B x F]
        blstats = blstats.view(T * B, -1).float()
        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == T * B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(T * B, -1)
        assert crop_rep.shape[0] == T * B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)  # -- TODO: slow?
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(T * B, -1)

        assert glyphs_rep.shape[0] == T * B

        # -- [B x K'']
        reps.append(glyphs_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        if self.use_lstm:
            core_input = st.view(T, B, -1)
            core_output_list = []
            core_output, core_state = self.rnn(core_input.squeeze(0), core_state, masks)
        else:
            core_output = st

        return core_output, core_state

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.baseline(core_features)

    def act(self, env_outputs, core_state, masks, deterministic=False):

        core_output, core_state = self._forward_base(env_outputs, core_state, masks, deterministic)

        # -- [B x A]
        dist = self.policy(core_output)
        # -- [B x A]
        value = self.baseline(core_output)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_dist, core_state

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.policy(core_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        value = self.baseline(core_features)
        return value, action_log_probs, dist_entropy, rnn_hxs