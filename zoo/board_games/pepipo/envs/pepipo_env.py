from typing import List, Optional
import copy

from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces
import numpy as np

from game import Game, Board, t_Piece, Piece

"""
# New obs space
- Box(0, )

# New board representation
- np.zeros(shape=(64,), dytpe=np.int8) (can use np.reshape to turn back to matrix)
- odd = p1, even = p2
0 | empty
1 | p1 pe
2 | p2 pe
3 | p1 po
4 | p2 po
5 | p1 pi in p1 pe
6 | p2 pi in p2 pe
7 | p1 pi in p2 pe
8 | p2 pi in p1 pe

"""

@ENV_REGISTRY.register('pepipo')
class PePiPoEnv(BaseEnv):

    config = dict(
        # env_id (str): The name of the environment. (I think it goes to gym registry)
        # env_id="PePiPo",
        # battle_mode (str): The mode of the battle. Choices are 'self_play_mode' or 'alpha_beta_pruning'.
        battle_mode='self_play_mode',
        # battle_mode_in_simulation_env (str): The mode of Monte Carlo Tree Search. This is only used in AlphaZero.
        battle_mode_in_simulation_env='self_play_mode',
        # bot_action_type (str): The type of action the bot should take. Choices are 'v0' or 'alpha_beta_pruning'.
        bot_action_type='v0',
        # replay_path (str): The folder path where replay video saved, if None, will not save replay video.
        replay_path=None,
        # agent_vs_human (bool): If True, the agent will play against a human.
        agent_vs_human=False,
        # prob_random_agent (int): The probability of the random agent.
        prob_random_agent=0,
        # prob_expert_agent (int): The probability of the expert agent.
        prob_expert_agent=0,
        # channel_last (bool): If True, the channel will be the last dimension.
        channel_last=False,
        # scale (bool): If True, the pixel values will be scaled.
        scale=False,
        # stop_value (int): The value to stop the game.
        stop_value=1,
        # alphazero_mcts_ctree (bool): If True, the Monte Carlo Tree Search from AlphaZero is used.
        alphazero_mcts_ctree=False,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg


    def __init__(self, cfg=None):
        self.game = Game()
        self.cfg = cfg
        self.battle_mode = cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = 'self_play_mode'
        self.board_size = self.game.board.board_size

        # Set some randomness for selecting action.
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'

        self.players = [1, 2]
        self._current_player = 1
        self.env = None # PePiPoGymEnv() ???

        valid_piece_types = (t_Piece.PE, t_Piece.PI, t_Piece.PO)
        total_spots_on_board = self.game.board.board_size**2

        self.total_num_actions = len(valid_piece_types)*total_spots_on_board


    def reset(self, start_player_index: int = 0, init_state: Optional[np.ndarray] = None) -> dict:
        self.players = [1, 2]
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]

        self._action_space = spaces.Discrete(self.total_num_actions)
        self._reward_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # could be np.int8
        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(self.total_num_actions,), dtype=np.int8),
                "current_player_index": spaces.Discrete(2),
                "to_play": spaces.Discrete(2),
                # "board": spaces.Box(low=0, ???)
            }
        )

        return self.observe()


    def observe(self) -> dict:
        if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
            return {
                "observation": self._get_obs(),
                "action_mask": self._get_action_mask(),
                "current_player_index": self.players.index(self._current_player),
                "to_play": -1,
                # "board": copy.deepcopy(self.board),
            }
        elif self.battle_mode == 'self_play_mode':
            return {
                "observation": self._get_obs(),
                "action_mask": self._get_action_mask(),
                "current_player_index": self.players.index(self._current_player),
                "to_play": self._current_player,
                # "board": copy.deepcopy(self.board),
            }
        else:
            raise Exception(f"Hit unknown condition in observe() | battle_mode: {self.battle_mode}")


    def step(self, action) -> BaseEnvTimestep:
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action()
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.bot_action()

            flag = "agent"
            timestep = self._player_step(action, flag)

            if timestep.done:
                # The ``eval_episode_return`` is calculated from player 1's perspective.
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs[
                                                                               'to_play'] == 1 else timestep.reward

            return timestep
        return BaseEnvTimestep({}, 0, 0, {})


    def legal_actions(self) -> List[int]:
        agent = f"player_{self._current_player}"
        mask = np.zeros(shape=(self.total_num_actions), dtype=np.int8)
        # iterate through all possible actions
        # and mark 1 if legal and 0 if illegal
        for action in range(self.total_num_actions):
            piece_type, x, y = self.parse_piece_from_action(action)
            mask[action] = 1 if self.game.validate_move(x, y, piece_type, agent) else 0
        return mask


    def bot_action(self) -> int:
        return


    def random_action(self) -> int:
        return


    def render(self, mode="human") -> None:
        return None


    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space


    @property
    def action_mask(self) -> spaces.Space:
        return self._action_space


    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space


    def _get_obs(self) -> np.ndarray:
        return


    def _get_action_mask(self) -> np.ndarray:
        return


    def parse_piece_from_action(self, action) -> tuple[t_Piece, int, int]:
        # this is gross but whatever
        # TODO: unit test?
        piece_type = -1
        if action > -1 and action < 64: # 0 to 63 inclusively
            piece_type = t_Piece.PI
        elif action > 63 and action <= 127: # 64 to 127 inclusively
            piece_type = t_Piece.PE
        elif action > 127 and action <= 191: # 128 to 192 inclusively
            piece_type = t_Piece.PO

        # calc board coords
        indx = action % 64
        x = indx // self.game.board.board_size
        y = indx % self.game.board.board_size
        return piece_type, x, y


    def __repr__(self) -> str:
        return "LightZero PePiPo Env"

