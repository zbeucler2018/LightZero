from typing import List, Optional
import copy

from ding.envs.env.base_env import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from gymnasium import spaces
import numpy as np

from game import Game, Board, t_Piece, Piece
from zoo.board_games.pepipo.envs.mmab import AlphaBetaPruningBot
from zoo.board_games.pepipo.envs.mcts import MCTSBot


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

        self.bot_action_type = cfg.bot_action_type


        if self.bot_action_type == 'alpha_beta_pruning':
            self.alpha_beta_pruning_player = AlphaBetaPruningBot(self, cfg, 'alpha_beta_pruning_player')
        if self.bot_action_type == 'mcts':
            cfg_temp = EasyDict(cfg.copy())
            cfg_temp.save_replay = False
            cfg_temp.bot_action_type = None
            env_mcts = PePiPoEnv(EasyDict(cfg_temp))
            self.mcts_bot = MCTSBot(env_mcts, 'mcts_player', 50)

        # Set some randomness for selecting action.
        self.prob_random_agent = cfg.prob_random_agent
        self.prob_expert_agent = cfg.prob_expert_agent
        assert (self.prob_random_agent >= 0 and self.prob_expert_agent == 0) or (
                self.prob_random_agent == 0 and self.prob_expert_agent >= 0), \
            f'self.prob_random_agent:{self.prob_random_agent}, self.prob_expert_agent:{self.prob_expert_agent}'

        self.render_mode = cfg.render_mode

        self.players = ["player_1", "player_2"]
        self._current_player = "player_1"
        self._env = self

        valid_piece_types = (t_Piece.PE, t_Piece.PI, t_Piece.PO)
        total_spots_on_board = self.game.board.board_size**2

        self.total_num_actions = len(valid_piece_types)*total_spots_on_board


    def reset(self, start_player_index: int = 0, init_state: Optional[np.ndarray] = None) -> dict:
        if init_state is not None:
            init_board = np.array(copy.deepcopy(init_state.reshape((self.board_size, self.board_size))))
            self.game.board = self.convert_board_to_state(init_board)
        else:
            self.game.board.empty_board()

        self.players = ["player_1", "player_2"]
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
                "board": spaces.Box(low=0, high=8, shape=(self.board_size**2,), dtype=np.int8)
            }
        )
        return self.observe()


    def mmab_reset(self, start_player_index=0, init_state=None):
        self.start_player_index = start_player_index
        self._current_player = self.players[self.start_player_index]
        if init_state is not None:
            self.game.board = self.convert_board_to_state(init_state.reshape((self.board_size, self.board_size)))
        else:
            self.env.game.board.empty_board()


    def mmab_simulate_action(self, board, start_player_index, action) -> tuple:
        self.reset(start_player_index, init_state=board) # reset the env metadata, not the board
        if action not in self.legal_actions():
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))
        # execute action
        t_piece, x, y = self.parse_piece_from_action(action)
        self.game.make_move(x, y, t_piece, self._current_player)
        # generate new legal actions and new board
        new_legal_actions = copy.deepcopy(self.legal_actions())
        new_board = copy.deepcopy(self.board)
        return new_board, new_legal_actions


    def mcts_simulate_action(self, action: int):
        """
        Overview:
            execute action and get next_simulator_env. used in AlphaZero.
        Arguments:
            - action: an integer from the action space.
        Returns:
            - next_simulator_env: next simulator env after execute action.
        """
        if action not in self.legal_actions():
            raise ValueError("action {0} on board {1} is not legal".format(action, self.board))

        new_board = copy.deepcopy(self.board)

        t_piece, x, y = self.parse_piece_from_action(action)
        self.game.make_move(x, y, t_piece, self._current_player)

        start_player_index = 1 if self.start_player_index == 0 else 0

        next_simulator_env = copy.deepcopy(self)
        next_simulator_env.reset(start_player_index, init_state=new_board)
        return next_simulator_env


    def observe(self) -> dict:
        if self.battle_mode == 'play_with_bot_mode' or self.battle_mode == 'eval_mode':
            return {
                "observation": self._get_obs(),
                "action_mask": self._get_action_mask(),
                "current_player_index": self.players.index(self._current_player),
                "to_play": -1,
                "board": copy.deepcopy(self.board),
            }
        elif self.battle_mode == 'self_play_mode':
            return {
                "observation": self._get_obs(),
                "action_mask": self._get_action_mask(),
                "current_player_index": self.players.index(self._current_player),
                "to_play": self._current_player,
                "board": copy.deepcopy(self.board),
            }
        else:
            raise Exception(f"Hit unknown condition in observe() | battle_mode: {self.battle_mode}")


    def modify_state(self, action) -> BaseEnvTimestep:
        # 1. validate move
        t_piece, x, y = self.parse_piece_from_action(action)
        assert self.game.validate_move(x, y, t_piece, self._current_player), f"The following move is invalid for {self._current_player}: {t_piece} ({x}, {y})"

        # 2. make move
        self.game.make_move(x, y, t_piece, self._current_player)

        # 3. check winner and get reward
        reward = np.array(0).astype(np.float32)
        done = False
        if self.game.check_winner(self._current_player):
            reward = np.array(1).astype(np.float32)
            done = True
        elif self.game.check_tie(self._current_player):
            done = True

        # 4. rotate player
        self._current_player = self._next_player

        # 5. populate BaseEnvTimestep with obs, done, etc
        info = {}
        if done:
            info['eval_episode_return'] = reward
        obs = self.observe()
        timestep = BaseEnvTimestep(obs, reward, done, info)

        self.render()

        return timestep


    def step(self, action) -> BaseEnvTimestep:
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
                # The ``eval_episode_return`` is calculated from player 1's perspective.
                timestep.info['eval_episode_return'] = -timestep.reward if timestep.obs['to_play'] == 1 else timestep.reward
            return timestep

        # 2 bots face each other
        if self.battle_mode == "play_with_bot_mode":
            # player1's turn
            p1_timestep = self.modify_state(action)
            if p1_timestep.done:
                # NOTE: in ``play_with_bot_mode``, we must set to_play as -1, because we don't consider the alternation between players.
                # And the ``to_play`` is used in MCTS.
                p1_timestep.obs['to_play'] = -1
                return p1_timestep

            # player2's turn
            bot_action = self.bot_action()
            p2_timestep = self.modify_state(bot_action)
            # NOTE: I dont really understand below (calcs player2 total episode reward?)
            # The ``eval_episode_return`` is calculated from player 1's perspective.
            p2_timestep.info['eval_episode_return'] = -p2_timestep.reward
            p2_timestep = p2_timestep._replace(reward=-p2_timestep.reward)
            # NOTE: in ``play_with_bot_mode``, we must set to_play as -1, because we don't consider the alternation between players.
            # And the ``to_play`` is used in MCTS.
            p2_timestep.obs['to_play'] = -1
            return p2_timestep

        # bot (p1) vs human (p2)
        if self.battle_mode == "eval_mode":
            # player 1's turn
            p1_timestep = self.modify_state(action)
            if p1_timestep.done:
                # NOTE: in ``eval_mode``, we must set to_play as -1, because we don't consider the alternation between players.
                # And the ``to_play`` is used in MCTS.
                p1_timestep.obs['to_play'] = -1
                return p1_timestep

            # player2's turn
            bot_action = self.human_to_action() if self.cfg.agent_vs_human else self.bot_action()
            p2_timestep = self.modify_state(bot_action)
            # NOTE: I dont really understand below (calcs player2 total episode reward?)
            # The 'eval_episode_return' is calculated from player1's perspective
            p2_timestep.info['eval_episode_return'] = -p2_timestep.reward
            p2_timestep = p2_timestep._replace(reward=-p2_timestep.reward)
            # NOTE: in ``eval_mode``, we must set to_play as -1, because we don't consider the alternation between players.
            # And the ``to_play`` is used in MCTS.
            p2_timestep.obs['to_play'] = -1
            return p2_timestep


    def legal_actions(self) -> List[int]:
        '''Returns a list of legal moves for the current agent'''
        legal_actions = []
        agent = self._current_player
        for action in range(self.total_num_actions):
            piece_type, x, y = self.parse_piece_from_action(action)
            if self.game.validate_move(x, y, piece_type, agent):
                legal_actions.append(action)
        return legal_actions


    def bot_action(self) -> int:
        if self.bot_action_type == "random":
            return self.random_action()
        elif self.bot_action_type == "alpha_beta_pruning":
            indx = self.players.index(self._current_player)
            return self.alpha_beta_pruning_player.get_best_action(self.board, player_index=indx)
        elif self.bot_action_type == "mcts":
            indx = self.players.index(self._current_player)
            return self.mcts_bot.get_actions(self.board, player_index=indx)
        else:
            raise NotImplementedError(f"The bot_action_type: {self.bot_action_type} is not implimented")


    def random_action(self) -> int:
        '''Sample a random legal action'''
        action_list = self.legal_actions()
        return np.random.choice(action_list)


    def _get_obs(self, player=None) -> np.ndarray:
        """Generates the observation from the state (board). ONLY WORKS FOR 2 PLAYERS"""
        # All possible states of a spot on the board
        # 0. empty
        # 1. my pe or my pi and my pe
        # 2. op pe or op pi and my pe
        # 3. my po
        # 4. op po
        # 5. my pi in op pe
        # 6. op pi in my pe
        if player is None:
            player = self._current_player
        n_possible_cell_states = 6
        base = np.zeros(shape=(self.game.board.board_size, self.game.board.board_size), dtype=np.float16)
        for x in range(self.game.board.board_size):
            for y in range(self.game.board.board_size):
                cell = self.game.board[x, y]

                # my pe
                if cell[1].player_id == player and cell[1]._typename == t_Piece.PE:
                    base[x, y] = 1

                # op pe
                if cell[1].player_id != player and cell[1]._typename == t_Piece.PE:
                    base[x, y] = 2

                # my po or my pe+pi
                if (cell[1].player_id == player and cell[1]._typename == t_Piece.PO) or \
                    ((cell[0].player_id == player and cell[0]._typename == t_Piece.PI) and (cell[1].player_id == player and cell[1]._typename == t_Piece.PE)):
                    base[x, y] = 3

                # op po or op pe+pi
                if (cell[1].player_id != player and cell[1]._typename == t_Piece.PO) or \
                    ((cell[0].player_id != player and cell[0]._typename == t_Piece.PI) and (cell[1].player_id != player and cell[1]._typename == t_Piece.PE)):
                    base[x, y] = 4

                # my pi in op pe
                if (cell[0].player_id == player and cell[0]._typename == t_Piece.PI) and (cell[1].player_id != player and cell[1]._typename == t_Piece.PE):
                    base[x, y] = 5

                # op pi in my pe
                if (cell[0].player_id != player and cell[0]._typename == t_Piece.PI) and (cell[1].player_id == player and cell[1]._typename == t_Piece.PE):
                    base[x, y] = 6

        # normalize
        base = base / n_possible_cell_states
        return base


    def _get_action_mask(self, player=None) -> np.ndarray:
        if player is None:
            player = self._current_player
        mask = np.zeros(shape=(3*64), dtype=np.int8)
        # iterate through all possible actions
        # and mark 1 if legal and 0 if illegal
        for action in range(3*64):
            piece_type, x, y = self.parse_piece_from_action(action)
            mask[action] = 1 if self.game.validate_move(x, y, piece_type, player) else 0
        return mask


    def parse_piece_from_action(self, action) -> tuple[t_Piece, int, int]:
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


    def convert_piece_and_coordinates_to_action(self, piece_type: str, x: int, y: int) -> int:
        # Determine the base action value based on piece type
        if piece_type == "pi": # t_Piece.PI:
            base_action = 0
        elif piece_type == "pe": # t_Piece.PE:
            base_action = 64
        elif piece_type == "po": # t_Piece.PO:
            base_action = 128
        else:
            raise ValueError("Invalid piece type {piece_type}")

        # Calculate the action from coordinates
        indx = x * self.board_size + y
        action = base_action + indx
        return action


    def human_to_action(self) -> int:
        '''Get an action from the human player'''
        while True:
            try:
                user_input = input("Enter PE, PI, or PO: ").lower()
                user_x = int(input("Enter x: "))
                user_y = int(input("Enter y: "))
                return self.convert_piece_and_coordinates_to_action(user_input, user_x, user_y)
            except ValueError as e:
                continue
            except Exception as e:
                if not isinstance(e, ValueError):
                    raise e


    def render(self, mode="ascii") -> None:
        if self.render_mode == 'ascii':
            self.game.print_board()


    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)


    def close(self):
        pass


    def convert_board_to_state(self, board: np.ndarray) -> Board:
        '''Converts a numpy array representation of a board into a Board class representation'''
        tmp = Board()
        for y in range(self.board_size):
            for x in range(self.board_size):
                match board[x,y]:
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
                        raise Exception(f"Got unknown board spot ({x}, {y}) {self.board[x, y]}")
        return tmp


    @property
    def _next_player(self) -> str:
        return "player_2" if self._current_player == "player_1" else "player_1"


    @property
    def board(self) -> np.ndarray:
        """
        These envs need a 'global' view of the board and Game.board isn't really right for this
        - odd = p1, even = p2
        - could condense 5 and 6 as 3 and 4 respectively
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
        tmp_board = np.zeros(shape=(64,), dtype=np.int8)
        for indx, spot in enumerate(self.game.board.board):
            left, right = spot

            if left._typename == t_Piece.EMPTY and right._typename == t_Piece.EMPTY:
                tmp_board[indx] = 0
            if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PE and right.player_id == "player_1"):
                tmp_board[indx] = 1
            if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PE and right.player_id == "player_2"):
                tmp_board[indx] = 2
            if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PO and right.player_id == "player_1"):
                tmp_board[indx] = 3
            if left._typename == t_Piece.EMPTY and (right._typename == t_Piece.PO and right.player_id == "player_2"):
                tmp_board[indx] = 4
            if (left._typename == t_Piece.PI and left.player_id == "player_1") and (right._typename == t_Piece.PE and right.player_id == "player_1"):
                tmp_board[indx] = 5
            if (left._typename == t_Piece.PI and left.player_id == "player_2") and (right._typename == t_Piece.PE and right.player_id == "player_2"):
                tmp_board[indx] = 6
            if (left._typename == t_Piece.PI and left.player_id == "player_1") and (right._typename == t_Piece.PE and right.player_id == "player_2"):
                tmp_board[indx] = 7
            if (left._typename == t_Piece.PI and left.player_id == "player_2") and (right._typename == t_Piece.PE and right.player_id == "player_1"):
                tmp_board[indx] = 8

        return tmp_board


    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space


    @property
    def action_space(self) -> spaces.Space:
        return self._action_space


    @property
    def reward_space(self) -> spaces.Space:
        return self._reward_space


    def __repr__(self) -> str:
        return "LightZero PePiPo Env"
