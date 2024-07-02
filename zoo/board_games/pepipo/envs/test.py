# from pepipo_env import PePiPoEnv
# from easydict import EasyDict
# from game import t_Piece, Board

# from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv

# # @pytest.fixture
# # def env():
# #     return PePiPoEnv()

# # def test_random_game(env: PePiPoEnv):
#     # 2 random bots play eachother

# if True:
#     random_agents_faceoff_cfg = EasyDict(dict(
#         battle_mode='play_with_bot_mode', # play_with_bot_mode self_play_mode eval_mode
#         agent_vs_human=False,
#         bot_action_type='random',
#         prob_random_agent=0,
#         prob_expert_agent=0,
#         render_mode='ascii'
#     ))

#     random_agent_vs_human = EasyDict(dict(
#         battle_mode='eval_mode', # play_with_bot_mode self_play_mode eval_mode
#         agent_vs_human=True,
#         bot_action_type='random',
#         prob_random_agent=0,
#         prob_expert_agent=0,
#         render_mode='ascii'
#     ))

#     mm_vs_human = EasyDict(dict(
#         battle_mode='eval_mode',
#         agent_vs_human=True,
#         bot_action_type='alpha_beta_pruning',
#         prob_random_agent=0,
#         prob_expert_agent=0,
#         render_mode='ascii'
#     ))

#     mcts_vs_human = EasyDict(dict(
#         battle_mode='eval_mode',
#         agent_vs_human=True,
#         bot_action_type='mcts',
#         prob_random_agent=0,
#         prob_expert_agent=0,
#         render_mode='ascii'
#     ))

#     ttt_mm_vs_human = EasyDict(dict(
#         battle_mode='eval_mode',
#         agent_vs_human=True,
#         bot_action_type='alpha_beta_pruning',
#         prob_random_agent=0,
#         prob_expert_agent=0,
#         channel_last=False,
#         scale=False,
#         alphazero_mcts_ctree=False
#     ))


#     cfg = mcts_vs_human

#     env = PePiPoEnv(cfg)
#     # env = TicTacToeEnv(ttt_mm_vs_human)

#     obs = env.reset()

#     env.render()

#     print("**** START ****")

#     while True:
#         action = env.bot_action()
#         # print(*env.parse_piece_from_action(action))
#         # print(env.game.pos_per_player)

#         timestep = env.step(action)

#         if timestep.done:
#             print(f"{env._current_player}({timestep.reward}) won!")
#             env.render()
#             break

#     env.close()



class idk:
    def __init__(self, a, b):
        self._a = a
        self._b = b

    @property
    def c(self):
        return 3

    @c.setter
    def c(self, value):
        self._c = value


i = idk(1, 2)
print(i._a, i._b)

print(i.c)

i.c = 5

print(i.c)
