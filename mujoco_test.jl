# using Revise
# using MPOPIS
using PyCall

# run(`$(PyCall.python) -m pip install dm_control`)
# run(`$(PyCall.python) -m pip install envpool`)
# run(`$(PyCall.python) -m pip install pip --upgrade`)
# run(`$(PyCall.python) -m pip install gym`)
# run(`$(PyCall.python) -m pip install imageio`)
# run(`$(PyCall.python) -m pip install brax`)

py"""
# from dm_control import mujoco
# from dm_control import suite
import envpool
import gym

# max_len = max(len(d) for d, _ in suite.BENCHMARKING)
# for domain, task in suite.BENCHMARKING:
    # print(f'{domain:<{max_len}}  {task}')

# def get_env(domain_name, task, random_seed=42):
    # random_state = np.random.RandomState(random_seed)
    # return suite.load(domain_name, task, task_kwargs={'random': random_state})

def get_envs_ep(env_name, env_type, num_envs, frame_skip, seeds=42, noise=0.0):
    return envpool.make(
        env_name,
        env_type=env_type,
        num_envs=num_envs,
        seed=seeds,
        reset_noise_scale=noise,
        frame_skip=frame_skip,
        gym_reset_return_info=True
    )

# def get_envs(env_name, num_env):
#     return gym.vector.make(env_name, num_envs=num_env)

# def reset_all(envs, seed):
#     env_data = envs.reset(seed = [seed for _ in range(envs.num_envs)])
#     return env_data

# duration = 4  # Seconds
# frames = []
# ticks = []
# rewards = []
# observations = []

# spec = env.action_spec()
# time_step = env.reset()

# while env.physics.data.time < duration:

#     action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
#     time_step = env.step(action)

#     camera0 = env.physics.render(camera_id=0, height=200, width=200)
#     camera1 = env.physics.render(camera_id=1, height=200, width=200)
#     frames.append(np.hstack((camera0, camera1)))
#     rewards.append(time_step.reward)
#     observations.append(copy.deepcopy(time_step.observation))
#     ticks.append(env.physics.data.time)

"""

function get_envs(env_name, num_envs; frame_skip=5)
    return py"get_envs_ep"(env_name, "gym", num_envs, frame_skip)
end

# function simulate_env(env, steps, acts)
#     rewards = zeros(steps)
#     for ii in 1:steps
#         ts = env.step(acts[ii, :])
#         if isnothing(ts[2])
#             rewards[ii] = 0.0
#         else
#             rewards[ii] = ts[2]
#         end
#     end
#     return rewards
# end

# function test_run()

#     env = py"get_env"("swimmer", "swimmer6")

#     ts = env.reset()

#     for i = 1:10
#         acts = rand(5) .*2 .- 1
#         ts = env.step(acts)
#     end
#     return env
# end

# test_run()

# py"""
# envs = gym.vector.make('CartPole-v1', num_envs=3)
# """