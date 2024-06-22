from pepipo_env import PePiPoEnv
from easydict import EasyDict
import pytest
from game import t_Piece, Board

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
        battle_mode='self_play_mode', # play_with_bot_mode self_play_mode eval_mode
        agent_vs_human=True,
        bot_action_type='random',
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode='ascii'
    ))

    mm_vs_human = EasyDict(dict(
        battle_mode='self_play_mode',
        agent_vs_human=True,
        bot_action_type='random',
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode='ascii'
    ))

    cfg = random_agent_vs_human

    env = PePiPoEnv(cfg)

    obs = env.reset()

    env.render()

    p =0
    while True:
        p+=1
        action = env.random_action()
        print(*env.parse_piece_from_action(action))
        print(env.game.pos_per_player)
        print(p)
        timestep = env.step(action)


        if p % 10 == 0:
            env.game.board = Board()


        if timestep.done:
            print(f"{env._current_player}({timestep.reward}) won!")
            env.render()
            break

    env.close()
