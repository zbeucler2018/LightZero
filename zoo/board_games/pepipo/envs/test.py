from pepipo_env import PePiPoEnv
from easydict import EasyDict
import pytest

# @pytest.fixture
# def env():
#     return PePiPoEnv()

# def test_random_game(env: PePiPoEnv):
    # 2 random bots play eachother

if True:
    cfg = EasyDict(dict(
        battle_mode='play_with_bot_mode',
        bot_action_type='random',
        prob_random_agent=0,
        prob_expert_agent=0,
        render_mode='ascii'
    ))

    env = PePiPoEnv(cfg)

    obs = env.reset()

    env.render()

    while True:
        action = env.random_action()
        # print(*env.parse_piece_from_action(action))
        timestep = env.step(action)
        print(timestep)
        # env.render()

        if timestep.done:
            print(f"{env._current_player}({timestep.reward}) won!")
            env.render()
            break

    env.close()
