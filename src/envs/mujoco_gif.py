
import gym
import numpy as np
import imageio

def make_gif(env_name, acts, gif_name, noise_scale, frame_skip):
    if env_name == 'Ant-v4':
        env = gym.make(
            env_name, 
            reset_noise_scale=noise_scale,
            frame_skip=frame_skip,
            terminate_when_unhealthy=False,
            render_mode='rgb_array'
        )
    else:
        env = gym.make(
            env_name, 
            reset_noise_scale=noise_scale,
            frame_skip=frame_skip,
            render_mode='rgb_array'
        )

    act_limit = env.action_space.high[0]
    o = env.reset()

    # For video / GIF.
    ep_obs = []
    dur = 0.01 * frame_skip

    print(acts.shape)
    # for _ in range(len(acts)):
    #     # obs = env.render(width=width, height=height)
    #     obs = env.render()
    #     # assert obs.shape == (height, width, 3), obs.shape  # height first!
    #     ep_obs.append(obs)

    #     # Take action, step into environment, etc.
    #     a = np.random.randn(act_dim)
    #     a = np.clip(a, -act_limit, act_limit)
    #     o, r, _, _, _ = env.step(a)


    # ep_name = f'ep_{env_name}_dur_{dur}_len_{str(ep_len).zfill(3)}.gif'
    # with imageio.get_writer(ep_name, mode='I', duration=dur) as writer:
    #     for obs_np in ep_obs:
    #         writer.append_data(obs_np)

