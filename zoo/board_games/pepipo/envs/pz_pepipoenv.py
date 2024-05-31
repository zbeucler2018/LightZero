from game import Game, t_Piece, Piece, Colors

from gymnasium import spaces
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import pygame


PLAYER_COLOR_MAP = {
   "player_0": (0, 255, 0),
   "player_1": (0, 0, 255),
   "player_2": (255, 0, 0),
   "player_3": (255, 0, 255)
}

class PePiPoEnv(AECEnv):

    metadata = {
        "name": "pepipo_0v",
        "is_parallelizable": False,
        "render_fps": 10,
        "render_modes": ["human", "ascii"],
    }

    def __init__(self, render_mode: str = "ascii", verbose: bool = False) -> None:
        super().__init__()
        self.game: Game = Game()

        # AEC API
        self.agents = [f"player_{p}" for p in range(self.game.n_players)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.rewards = {i: 0 for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        valid_piece_types = [t_Piece.PE, t_Piece.PI, t_Piece.PO]
        total_spots_on_board = self.game.board.board_size * self.game.board.board_size

        # action space
        # 3 pieces actions (PE, PI, PO) * 64 possible spots on the board (8x8 board)
        self.action_spaces = {i: spaces.Discrete(len(valid_piece_types)*total_spots_on_board) for i in self.agents}

        # obs space
        # 8x8x4
        # Not sure why I have to make the obs space a dict, this was how the connect_four env is (and other classic pz envs)
        self.observation_spaces = {
            i: spaces.Dict({
                "observation": spaces.Box(low=0, high=1, shape=(self.game.board.board_size, self.game.board.board_size), dtype=np.float16),
                "action_mask": spaces.Box(low=0, high=1, shape=(len(valid_piece_types)*total_spots_on_board,), dtype=np.int8)
            }) for i in self.agents
        }

        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((800, 800))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("PePiPo")

    def parse_piece_from_action(self, action: int) -> tuple[t_Piece, int, int]:
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

    def observe(self, agent) -> dict:
        return {"observation": self._get_obs(agent), "action_mask": self._get_action_mask(agent)}
    
    def _get_obs(self, agent) -> np.ndarray:
        """Generates the observation from the state (board). ONLY WORKS FOR 2 PLAYERS"""
        # NOTE: ONLY WORKS WITH 2 PLAYERS
        # All possible states of a spot on the board
        # 0. empty
        # 1. my pe or my pi and my pe
        # 2. op pe or op pi and my pe
        # 3. my po
        # 4. op po
        # 5. my pi in op pe
        # 6. op pi in my pe
        n_possible_cell_states = 6
        base = np.zeros(shape=(self.game.board.board_size, self.game.board.board_size), dtype=np.float16)
        for x in range(self.game.board.board_size):
            for y in range(self.game.board.board_size):
                cell = self.game.board[x, y]

                # my pe
                if cell[1].player_id == agent and cell[1]._typename == t_Piece.PE:
                    base[x, y] = 1

                # op pe
                if cell[1].player_id != agent and cell[1]._typename == t_Piece.PE:
                    base[x, y] = 2
                
                # my po or my pe+pi
                if (cell[1].player_id == agent and cell[1]._typename == t_Piece.PO) or \
                    ((cell[0].player_id == agent and cell[0]._typename == t_Piece.PI) and (cell[1].player_id == agent and cell[1]._typename == t_Piece.PE)):
                    base[x, y] = 3
                
                # op po or op pe+pi
                if (cell[1].player_id != agent and cell[1]._typename == t_Piece.PO) or \
                    ((cell[0].player_id != agent and cell[0]._typename == t_Piece.PI) and (cell[1].player_id != agent and cell[1]._typename == t_Piece.PE)):
                    base[x, y] = 4
                
                # my pi in op pe
                if (cell[0].player_id == agent and cell[0]._typename == t_Piece.PI) and (cell[1].player_id != agent and cell[1]._typename == t_Piece.PE):
                    base[x, y] = 5

                # op pi in my pe
                if (cell[0].player_id != agent and cell[0]._typename == t_Piece.PI) and (cell[1].player_id == agent and cell[1]._typename == t_Piece.PE):
                    base[x, y] = 6

        # normalize
        base = base / n_possible_cell_states
        return base

    def observation_space(self, agent) -> np.ndarray:
        return self.observation_spaces[agent]

    def _get_action_mask(self, agent) -> np.ndarray:
        mask = np.zeros(shape=(3*64), dtype=np.int8)
        # iterate through all possible actions
        # and mark 1 if legal and 0 if illegal
        for action in range(3*64):
            piece_type, x, y = self.parse_piece_from_action(action)
            mask[action] = 1 if self.game.validate_move(x, y, piece_type, agent) else 0
        return mask

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            return self._was_dead_step(action)

        agent = self.agent_selection

        # self._cumulative_rewards[agent] = 0


        # decode action
        piece_type, x, y = self.parse_piece_from_action(action)

        # print(agent, piece_type, x, y)

        # validate move
        assert self.game.validate_move(x, y, piece_type, agent), f"Move {piece_type.name}({x}, {y}) for {agent} is invalid"

        # make move
        self.game.make_move(x, y, piece_type, agent)

        # check winner
        if self.game.check_winner(agent):
            self.rewards = {i: -1 for i in self.agents}
            self.rewards[self.agent_selection] = 1  # winner gets +1 reward, loser gets -1
            self.terminations = {i: True for i in self.agents}
            self.truncations = {i: True for i in self.agents}
        elif self.game.check_tie(agent):
            self.rewards = {i: 0 for i in self.agents} # 0 reward for all agents in a tie
            self.terminations = {i: True for i in self.agents}
            self.truncations = {i: True for i in self.agents}

        # Switch selection to next agents
        self.agent_selection = self._agent_selector.next()

        # Adds .rewards to ._cumulative_rewards
        # NOTE: I don't understand self._cumulative_rewards
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def reset(self, seed=None, options=None):
        self.game = Game() # resets board
        self.agents = self.possible_agents[:]
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.rewards = {i: 0 for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}

    def render(self) -> None:
        if self.render_mode == "ascii":
            self.game.print_board()
        elif self.render_mode == "human":
            # TODO: Add lines between each spot on the board
            # TODO: Display the current player (in their color?)
            # TODO: Display the winner
            # TODO: Display POs per player
            self.screen.fill((0,0,0))
            # pygame.draw.rect(self.screen, (139,69,19), pygame.Rect((100, 100), (800, 800))) # this will be the rendered 'board'


            def draw_board_spot(center_coords: tuple[int, int], spot: list[Piece, Piece]) -> None:
                """Draws the spot at the center coordinates"""
                if spot[0]._typename == t_Piece.EMPTY and spot[1]._typename == t_Piece.EMPTY:
                    return
                for s in spot:
                    if s._typename == t_Piece.PE:
                        pygame.draw.circle(self.screen, PLAYER_COLOR_MAP[s.player_id], center_coords, 40, 10)
                    elif s._typename == t_Piece.PI:
                        pygame.draw.circle(self.screen, PLAYER_COLOR_MAP[s.player_id], center_coords, 20)
                    elif s._typename == t_Piece.PO:
                        pygame.draw.circle(self.screen, PLAYER_COLOR_MAP[s.player_id], center_coords, 40)

            # x: [150:50:850], y: [150:50:850]
            for x in range(self.game.board.board_size):
                for y in range(self.game.board.board_size):
                    spot = self.game.board[x, y]
                    canvas_coords = ((x * 100) + 50, (y * 100) + 50)
                    draw_board_spot(canvas_coords, spot)

            pygame.display.flip()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self.render_mode == "human":
            pygame.quit()


if __name__ == "__main__":
    for i in range(4):
        env = PePiPoEnv(render_mode="human")

        # from pettingzoo.classic import tictactoe_v3
        # env = tictactoe_v3.env(render_mode="human")
        # from pettingzoo.butterfly import cooperative_pong_v5
        # env = cooperative_pong_v5.env(render_mode="human")

        env.reset(seed=42)

        t_steps = 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            print(agent, reward)
            t_steps += 1

            if termination or truncation:
                action = None
            else:
                # invalid action masking is optional and environment-dependent
                if "action_mask" in info:
                    mask = info["action_mask"]
                elif isinstance(observation, dict) and "action_mask" in observation:
                    mask = observation["action_mask"]
                else:
                    mask = None
                action = env.action_space(agent).sample(mask)

            env.step(action)
            env.render()

            if t_steps > 5_000: assert False, "Random game went above 5k moves so something is wrong"

        env.close()
