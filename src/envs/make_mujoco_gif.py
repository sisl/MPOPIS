
import gym
import numpy as np
import imageio
import argparse
 
 
parser = argparse.ArgumentParser()
parser.add_argument("-env", "--env_name", default="Ant-v4",
    help="Name of envrionment (e.g. 'Ant-v4')")
parser.add_argument("-af", "--acts_file", default=None,
    help="CSV containing actions")
parser.add_argument("-o", "--output_fname", default="mujoco_gif",
    help="Name of gif without the extension (e.g. output1)")

def main():
    args = parser.parse_args()

    acts = np.genfromtxt(args.acts_file, delimiter=',')

    if args.env_name == 'Ant-v4':
        env = gym.make(
            args.env_name, 
            reset_noise_scale=0.0,
            terminate_when_unhealthy=False,
            render_mode='rgb_array'
        )
    else:
        env = gym.make(
            args.env_name, 
            reset_noise_scale=0.0,
            render_mode='rgb_array'
        )
    assert acts.shape[1] == env.action_space.shape[0], "Action dim doesn't match"
    mujoco_frame_skip = 100 / env.metadata["render_fps"]
    dur = 0.01 * mujoco_frame_skip

    o = env.reset() 
    ep_obs = []
    
    obs = env.render()
    ep_obs.append(obs)
    rew = 0
    for act in acts:
        o, r, _, _, i = env.step(act)
        rew += r
        obs = env.render()
        ep_obs.append(obs)

    print(f'Total Reward: {rew:.4f}')
    with imageio.get_writer(args.output_fname + '.gif', mode='I', duration=dur) as writer:
        for obs_np in ep_obs:
            writer.append_data(obs_np)

if __name__ == "__main__":
    main()

