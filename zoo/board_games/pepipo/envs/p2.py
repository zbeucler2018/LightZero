from typing import List, Optional, Any
import copy
from sys import exit

from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces
import numpy as np

from game import Game, Board, t_Piece, Piece
from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot
from zoo.board_games.mcts_bot import MCTSBot


"""
**Using the tictactoe env as a reference**

- I should also use the dafault mmab and mcts bots if possible

# Spaces

|              | shape | type     | low | high |
| ------------ | ----- | -------- | --- | ---- |
| board        | (8,8) | int8     |  0  |  8   | NOTE: Should be a 'global' view of the state/board
| obs space    | (8,8) | box      |  0  |  8   | NOTE: Should be a _current_player specific view of the board
| action space | (192) | discrete |  0  |  192 |
| action mask  | (192) | int8     |  0  |  1   |
| reward space | (1)   | box      | -1  |  1   |
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
        # bot_action_type (str): The type of action the bot should take. Choices are 'random' or 'alpha_beta_pruning'.
        bot_action_type='random',
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
        # render_mode (str): 'human', 'ascii', or 'mp4'
        render_mode='ascii'
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg


    def __init__(self, cfg=None):
        self.cfg = cfg

        self.game = Game()

        # self.channel_last = cfg.channel_last
        # self.scale = cfg.scale

        self.battle_mode = cfg.battle_mode
        # The mode of interaction between the agent and the environment.
        assert self.battle_mode in ['self_play_mode', 'play_with_bot_mode', 'eval_mode']
        # The mode of MCTS is only used in AlphaZero.
        self.battle_mode_in_simulation_env = 'self_play_mode'

        self.board_size = 8
        self.board = np.zeros(shape=(self.board_size, self.board_size), dtype=np.int8)

        self.players = ["player_1", "player_2"]
        self._current_player = "player_1"
        self.start_player_index = 0

        self.valid_piece_types = (t_Piece.PE, t_Piece.PI, t_Piece.PO)
        self.total_spots_on_board = self.board_size**2
        self.total_num_actions = len(self.valid_piece_types)*self.total_spots_on_board

        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'

        self.agent_vs_human = cfg.agent_vs_human
        self.bot_action_type = cfg.bot_action_type
        self.alphazero_mcts_ctree = cfg.alphazero_mcts_ctree

        if self.bot_action_type == 'minimax':
            self.alpha_beta_pruning_player = AlphaBetaPruningBot(self, cfg, 'minimax_player')
        if self.bot_action_type == 'mcts':
            cfg_temp = EasyDict(cfg.copy())
            cfg_temp.save_replay = False
            cfg_temp.bot_action_type = None
            env_mcts = PePiPoEnv(EasyDict(cfg_temp))
            self.mcts_bot = MCTSBot(env_mcts, 'mcts_player', 50)


    def step(self, action: int) -> BaseEnvTimestep:
        '''Perform the action in the env'''
        # self play (training)
        if self.battle_mode == 'self_play_mode':
            if self.prob_random_agent > 0:
                if np.random.rand() < self.prob_random_agent:
                    action = self.random_action()
            elif self.prob_expert_agent > 0:
                if np.random.rand() < self.prob_expert_agent:
                    action = self.bot_action()
            timestep = self.modify_state(action)
            if timestep.done:
                # The `eval_episode_return` is calculated from player 1's perspective.
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs['to_play'] == 1 else timestep.reward
            return timestep

        # 2 bots face each other
        elif self.battle_mode == 'play_with_bot_mode':
            p1_timestep = self.modify_state(action)
            if p1_timestep.done:
                p1_timestep.obs['to_play'] = -1
                return p1_timestep

            bot_action = self.bot_action()
            p2_timestep = self.modify_state(bot_action)
            p2_timestep.info['eval_episode_return'] = -p2_timestep.reward
            p2_timestep = p2_timestep._replace(reward=-p2_timestep.reward)
            p2_timestep.obs['to_play'] = -1
            return p2_timestep

        # bot (p1) vs human or bot (p1)
        elif self.battle_mode == 'eval_mode':
            p1_timestep = self.modify_state(action)
            if p1_timestep.done:
                p1_timestep.obs['to_play'] = -1
                return p1_timestep

            bot_action = self.human_to_action() if self.agent_vs_human else self.bot_action()
            p2_timestep = self.modify_state(bot_action)
            p2_timestep.info['eval_episode_return'] = -p2_timestep.reward
            p2_timestep = p2_timestep._replace(reward=-p2_timestep.reward)
            p2_timestep.obs['to_play'] = -1
            return p2_timestep


    def reset(self, start_player_index=0, init_state=None, katago_policy_init=False, katago_game_state=None):
        '''Resets the enviroment'''
        # define spaces
        self._observation_space = spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size, 2), dtype=np.int8)
        self._action_space = spaces.Discrete(self.total_num_actions)
        self._reward_shape = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)

        # config players
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]

        # create board
        if init_state is not None:
            board = np.array(copy.deepcopy(init_state), dtype=np.int8)
            if self.alphazero_mcts_ctree:
                board = board.reshape((self.board_size, self.board_size))
            self.game.board = self.convert_numpy_array_to_board(board)
        else:
            self.game.board.empty_board()

        if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
            # In ``play_with_bot_mode`` and ``eval_mode``, we need to set the "to_play" parameter in the "obs" dict to -1,
            # because we don't take into account the alternation between players.
            # The "to_play" parameter is used in the MCTS algorithm.
            obs = {
                'observation': self._get_obs(),
                'action_mask': self._get_action_mask(),
                'board': copy.deepcopy(self.convert_board_to_numpy_array()),
                'current_player_index': self.start_player_index,
                'to_play': -1
            }
        elif self.battle_mode == 'self_play_mode':
            # In the "self_play_mode", we set to_play=self._current_player in the "obs" dict,
            # which is used to differentiate the alternation of 2 players in the game when calculating Q in the MCTS algorithm.
            obs = {
                'observation': self._get_obs(),
                'action_mask': self._get_action_mask(),
                'board': copy.deepcopy(self.convert_board_to_numpy_array()),
                'current_player_index': self.start_player_index,
                'to_play': self._current_player
            }
        return obs


    def reset_v2(self, start_player_index=0, init_state=None) -> None:
        '''Used by AlphaBetaPruningBot only'''
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.game.board = self.convert_numpy_array_to_board(init_state)
        else:
            self.game.board.empty_board()


    def simulate_action_v2(self, board, start_player_index, action):
        '''Used by AlphaBetaPruningBot only'''
        self.reset(start_player_index, init_state=board) # reset the env metadata, not the board
        if action not in self.legal_actions:
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        tpiece, x, y = self.parse_piece_from_action(action)
        self.game.make_move(x, y, tpiece, self._current_player)
        new_legal_actions = copy.deepcopy(self.legal_actions)
        new_board = copy.deepcopy(self.board)
        return new_board, new_legal_actions


    def modify_state(self, action: int) -> BaseEnvTimestep:
        # 1. validate move
        tpiece, x, y = self.parse_piece_from_action(action)
        assert self.game.validate_move(x, y, tpiece, self._current_player), \
            f"The following move is invalid for {self._current_player}: {tpiece} ({x}, {y})"

        # 2. make move
        self.game.make_move(x, y, tpiece, self._current_player)

        # 3. Get winner
        done, winner = self.get_done_winner()

        # 4. calculate reward and info
        reward = np.array(int(winner == self._current_player)).astype(np.int8)
        info = {' next player to play': self._next_player}

        # 5. Rotate player
        self._current_player = self._next_player

        # NOTE: As per docs, need to add this key if the game finishes
        if done:
            info['eval_episode_return'] = reward

        obs = {
            'observation': self._get_obs(),
            'action_mask': self._get_action_mask(),
            'board': copy.deepcopy(self.convert_board_to_numpy_array()),
            'current_player_index': self._current_player_index,
            'to_play': self._current_player,
        }

        return BaseEnvTimestep(obs, reward, done, info)


    def _get_obs(self, player=None) -> np.ndarray:
        '''Generates the observation for the given or current player'''
        # NOTE: In TicTacToeEnv, the obs is player specific and the board is global
        # This whole thing is gross
        # Dimentions
        # 1. The board only concerning the current player (0=empty,1=pe,2=pi,3=po,4=pi+po)
        # 2. The board only concerning the next player (0=empty,-1=pe,-2=pi,-3=po)
        # 3. the current player's index
        if player is None:
            player = self._current_player
        board = self.convert_board_to_numpy_array()
        d1_tmp = np.zeros(shape=(self.board_size, self.board_size), dtype=np.int8)
        d2_tmp = np.zeros(shape=(self.board_size, self.board_size), dtype=np.int8)
        for x in range(self.board_size):
            for y in range(self.board_size):
                match board[x, y]:
                    case 0: # empty
                        d1_tmp[x, y] = 0
                    case 1: # p1 pe
                        if self._current_player == "player_1":
                            d1_tmp[x, y] = 1
                    case 2: # p2 pe
                        if self._current_player == "player_2":
                            d2_tmp[x, y] = -1
                    case 3: # p1 po
                        if self._current_player == "player_1":
                            d1_tmp[x, y] = 3
                    case 4: # p2 po
                        if self._current_player == "player_2":
                            d2_tmp[x, y] = -3
                    case 5: # p1 pi in p1 pe
                        if self._current_player == "player_1":
                            d1_tmp[x, y] = 4
                    case 6: # p2 pi in p2 pe
                        if self._current_player == "player_2":
                            d2_tmp[x, y] = -4
                    case 7: # p1 pi in p2 pe (both players are here)
                        if self._current_player == "player_1":
                            d1_tmp[x, y] = 2
                            d2_tmp[x, y] = -1
                        elif self._current_player == "player_2":
                            d1_tmp[x, y] = -1
                            d2_tmp[x, y] = 2
                    case 8: # p2 pi in p1 pe (both players are here)
                        if self._current_player == "player_1":
                            d1_tmp = 1
                            d2_tmp = -2
                        elif self._current_player == "player_2":
                            d1_tmp = 2
                            d2_tmp = -1
        return np.array([d1_tmp, d2_tmp])


    def _get_action_mask(self, player=None) -> np.ndarray:
        '''Generates the action mask for the current or given player'''
        if player is None:
            player = self._current_player
        mask = np.zeros(shape=(self.total_num_actions), dtype=np.int8)
        for action in range(self.total_num_actions):
            tpiece, x, y = self.parse_piece_from_action(action)
            mask[action] = 1 if self.game.validate_move(x, y, tpiece, player) else 0
        return mask


    def convert_board_to_numpy_array(self, new_board: Optional[Board]=None) -> np.ndarray:
        '''Converts a numpy array to a Board class instance.
        - odd = p1, even = p2
        - could condense 5 and 6 as 3 and 4 respectively
        - 0 => empty
        - 1 => p1 pe
        - 2 => p2 pe
        - 3 => p1 po
        - 4 => p2 po
        - 5 => p1 pi in p1 pe
        - 6 => p2 pi in p2 pe
        - 7 => p1 pi in p2 pe
        - 8 => p2 pi in p1 pe
        '''
        if new_board is None:
            new_board = self.game.board
        tmp_board = np.zeros(shape=(self.board_size, self.board_size), dtype=np.int8)
        for y in range(self.board_size):
            for x in range(self.board_size):
                left, right = new_board[x, y]
                if left._typename == t_Piece.EMPTY and right._typename == t_Piece.EMPTY:
                    tmp_board[x, y] = 0
                if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PE and right.player_id == "player_1"):
                    tmp_board[x, y] = 1
                if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PE and right.player_id == "player_2"):
                    tmp_board[x, y] = 2
                if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PO and right.player_id == "player_1"):
                    tmp_board[x, y] = 3
                if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PO and right.player_id == "player_2"):
                    tmp_board[x, y] = 4
                if (left._typename == t_Piece.PI and left.player_id == "player_1") and (right._typename == t_Piece.PE and right.player_id == "player_1"):
                    tmp_board[x, y] = 5
                if (left._typename == t_Piece.PI and left.player_id == "player_2") and (right._typename == t_Piece.PE and right.player_id == "player_2"):
                    tmp_board[x, y] = 6
                if (left._typename == t_Piece.PI and left.player_id == "player_1") and (right._typename == t_Piece.PE and right.player_id == "player_2"):
                    tmp_board[x, y] = 7
                if (left._typename == t_Piece.PI and left.player_id == "player_2") and (right._typename == t_Piece.PE and right.player_id == "player_1"):
                    tmp_board[x, y] = 8
        return tmp_board


    def convert_numpy_array_to_board(self, new_board: np.ndarray) -> Board:
        '''Converts a Board instance to a numpy array'''
        tmp = Board()
        for y in range(self.board_size):
            for x in range(self.board_size):
                match new_board[x, y]:
                    case 0:
                        tmp[x, y] = [Piece(t_Piece.EMPTY), Piece(t_Piece.EMPTY)]
                    case 1:
                        tmp[x, y] = [Piece(t_Piece.EMPTY), Piece(t_Piece.PE, player_id="player_1")]
                    case 2:
                        tmp[x, y] = [Piece(t_Piece.EMPTY), Piece(t_Piece.PE, player_id="player_2")]
                    case 3:
                        tmp[x, y] = [Piece(t_Piece.EMPTY), Piece(t_Piece.PO, player_id="player_1")]
                    case 4:
                        tmp[x, y] = [Piece(t_Piece.EMPTY), Piece(t_Piece.PO, player_id="player_2")]
                    case 5:
                        tmp[x, y] = [Piece(t_Piece.PI, player_id="player_1"), Piece(t_Piece.PE, player_id="player_1")]
                    case 6:
                        tmp[x, y] = [Piece(t_Piece.PI, player_id="player_2"), Piece(t_Piece.PE, player_id="player_2")]
                    case 7:
                        tmp[x, y] = [Piece(t_Piece.PI, player_id="player_1"), Piece(t_Piece.PE, player_id="player_2")]
                    case 8:
                        tmp[x, y] = [Piece(t_Piece.PI, player_id="player_2"), Piece(t_Piece.PE, player_id="player_1")]
                    case _:
                        raise Exception(f"Got unknown board spot ({x}, {y}) {new_board[x, y]}")
        return tmp


    def parse_piece_from_action(self, action) -> tuple[t_Piece, int, int]:
        '''Converts a action to a piece type, x and y coordinates'''
        piece_type = -1
        if action > -1 and action < 64: # 0 to 63 inclusively
            piece_type = t_Piece.PI
        elif action > 63 and action <= 127: # 64 to 127 inclusively
            piece_type = t_Piece.PE
        elif action > 127 and action <= 191: # 128 to 192 inclusively
            piece_type = t_Piece.PO

        # calc board coords
        indx = action % 64
        x = indx // self.board_size
        y = indx % self.board_size
        return piece_type, x, y


    def parse_action_from_piece(self, tpiece: str, x: int, y: int) -> int:
        '''Calculates the action given the piece type, x and y coordinates'''
        base_action = x * self.board_size + y
        if tpiece == t_Piece.PI:
            action = base_action
        elif tpiece == t_Piece.PE:
            action = base_action + 64
        elif tpiece == t_Piece.PO:
            action = base_action + 128
        else:
            raise ValueError("Invalid piece type")
        return action


    def get_done_winner(self) -> tuple[bool, str]:
        '''Returns if the game is done and who won it'''
        # check wins
        if self.game.check_winner("player_1"):
            return True, "player_1"
        if self.game.check_winner("player_2"):
            return True, "player_2"
        # check tie
        if self.game.check_tie("player_1") or self.game.check_tie("player_2"):
            return True, -1
        # No wins or tie
        return False, -1


    def get_done_reward(self):
        # only rewards player1 if it won
        done, winner = self.get_done_winner()
        if winner == "player_1": # p1 w
            reward = 1
        elif winner == "player_2": # p2 w
            reward = -1
        elif winner == -1 and done: # tie
            reward = 0
        elif winner == -1 and not done:
            # episode is not done
            reward = None
        return done, reward


    def bot_action(self) -> int:
        if self.bot_action_type == 'random':
            return self.random_action()
        elif self.bot_action_type == 'minimax':
            return self.alpha_beta_pruning_player.get_best_action(self.board, self._current_player_index)
        elif self.bot_action_type == 'mcts':
            raise self.mcts_bot.get_actions(self.board, self._current_player_index)
        else:
            raise NotImplementedError


    def human_to_action(self) -> int:
        '''Asks the user to select a piece and coordinates for their move'''
        tpiece_to_string = {"pe": t_Piece.PE, "pi": t_Piece.PI, "po": t_Piece.PO}
        while True:
            try:
                user_input = input("Enter PE, PI, or PO: ").lower()
                user_x = int(input("Enter x: "))
                user_y = int(input("Enter y: "))
                choice = self.parse_action_from_piece(tpiece_to_string[user_input], user_x, user_y)
                if choice in self.legal_actions:
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("Quitting game...")
                exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice


    def random_action(self) -> int:
        '''Returns a valid random action for the currrent player'''
        return np.random.choice(self.legal_actions)


    def render(self, mode="ascii") -> None:
        if self.render_mode == 'ascii':
            self.game.print_board()


    @property
    def _next_player(self) -> str:
        return "player_2" if self._current_player == "player_1" else "player_1"


    @property
    def _current_player_index(self) -> int:
        return self.players.index(self._current_player)


    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space


    @property
    def action_space(self) -> spaces.Space:
        return self._action_space


    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space


    @property
    def legal_actions(self) -> list[int]:
        '''Returns a list of legal moves for the current player'''
        legal_actions = []
        for action in range(self.total_num_actions):
            piece_type, x, y = self.parse_piece_from_action(action)
            if self.game.validate_move(x, y, piece_type, self._current_player):
                legal_actions.append(action)
        return legal_actions


    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)


    def close(self):
        pass


    def __repr__(self) -> str:
        return "LightZero PePiPoEnv"
