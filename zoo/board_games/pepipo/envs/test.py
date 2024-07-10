# from pepipo_env import PePiPoEnv
from p2 import PePiPoEnv
from easydict import EasyDict
from game import t_Piece, Board

from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv

# @pytest.fixture
# def env():
#     return PePiPoEnv()

# def test_random_game(env: PePiPoEnv):
    # 2 random bots play eachother

if True:
    random_agents_faceoff_cfg = EasyDict(dict(
        battle_mode='play_with_bot_mode', # play_with_bot_mode self_play_mode eval_mode
        agent_vs_human=False,
        bot_action_type='random',
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode='ascii'
    ))

    random_agent_vs_human = EasyDict(dict(
        battle_mode='eval_mode', # play_with_bot_mode self_play_mode eval_mode
        agent_vs_human=True,
        bot_action_type='random',
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode='ascii'
    ))

    mm_vs_human = EasyDict(dict(
        battle_mode='eval_mode',
        agent_vs_human=True,
        bot_action_type='minimax',
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode='ascii',
        alphazero_mcts_ctree=False,
    ))

    mcts_vs_human = EasyDict(dict(
        battle_mode='eval_mode',
        agent_vs_human=True,
        bot_action_type='mcts',
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode='ascii',
        channel_last=False,
        alphazero_mcts_ctree=False
    ))

    ttt_mm_vs_human = EasyDict(dict(
        battle_mode='eval_mode',
        agent_vs_human=True,
        bot_action_type='alpha_beta_pruning',
        prob_random_agent=0,
        prob_expert_agent=0,
        channel_last=False,
        scale=False,
        alphazero_mcts_ctree=False,
        render_mode="human"
    ))


    cfg = mm_vs_human

    env = PePiPoEnv(cfg)
    # env = TicTacToeEnv(ttt_mm_vs_human)

    obs = env.reset()

    env.render()

    print("**** START ****")

    while True:
        action = env.bot_action()

        print(f"player: {env._current_player}", *env.parse_piece_from_action(action))
        # print(env.game.pos_per_player)

        timestep = env.step(action)

        if timestep.done:
            print(f"Game won! player_id={env._current_player} reward:{timestep.reward}")
            env.render()
            break

    env.close()
