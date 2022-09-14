using PyCall

mutable struct EnvpoolEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    const task::String
    py_env::PyObject
    const action_space::A
    const observation_space::Space{Vector{ClosedInterval{T}}}
    const num_states::Int
    const num_envs::Int
    info::Dict
    rews::Vector{T}
    state::Matrix{T}
    done::Bool
    t::Int
    acts::Vector{Vector{T}}
    rng::R
end

"""
EnvpoolEnv(task ;kwargs...)

# Keyword arguments

"""
function EnvpoolEnv(
    task = "Swimmer-v4";
    T = Float64,
    num_envs = 100,
    frame_skip = 10,
    random_seed = 42,
    rng = Random.GLOBAL_RNG,
)

    py"""
    import envpool
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
    """

    py_env = py"get_envs_ep"(task, "gym", num_envs, frame_skip)
    env_data = py_env.reset()
    py_action_space = py_env.action_space

    action_space = ClosedInterval{Vector{T}}(
        py_action_space.low,
        py_action_space.high,
    )

    py_observation_space = py_env.observation_space
    py_obs_len = py_observation_space.shape[1]
    py_obs_low = py_observation_space.low
    py_obs_high = py_observation_space.high

    observation_vec = [py_obs_low[ii]..py_obs_high[ii] for ii in 1:py_obs_len]
    observation_space = Space(observation_vec)

    env = EnvpoolEnv(
        task,
        py_env,
        action_space,
        observation_space,
        py_obs_len,
        num_envs,
        env_data[end],
        zeros(T, num_envs),
        env_data[1],
        false,
        Int(env_data[end]["elapsed_step"][1]),
        Vector{Vector{T}}(),
        rng,
    )

    reset!(env)
    env
end

Random.seed!(env::EnvpoolEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::EnvpoolEnv) = env.action_space
RLBase.state_space(env::EnvpoolEnv) = env.observation_space
RLBase.is_terminated(env::EnvpoolEnv) = env.done
RLBase.state(env::EnvpoolEnv) = env.state

RLBase.reward(env::EnvpoolEnv) = env.rews

function RLBase.reset!(env::EnvpoolEnv{A,T}; restore=false) where {A,T}
    env_data = env.py_env.reset()
    env.info = env_data[end]
    env.rews = zeros(T, env.num_envs)
    env.state = env_data[1]
    env.done = false
    env.t = Int(env_data[end]["elapsed_step"][1])
    if !restore
        env.acts = Vector{Vector{T}}()
    else
        _restore_using_acts!(env)
    end
    return env
end


function (env::EnvpoolEnv)(a::Vector)
    acts = repeat(a', env.num_envs)
    env(acts; update_acts=true)
end
function (env::EnvpoolEnv)(a::Matrix{Float64}; update_acts=false)
    if size(a)[2] == 1
        a = repeat(a', env.num_envs)
        update_acts = true
    end
    size(a)[1] == env.num_envs || error("Number of rows in action need to be num_envs")
    _step!(env, a)
    if update_acts
        push!(env.acts, vec(a[1, :]))
    end
    return env
end

function _step!(env::EnvpoolEnv, a::Matrix{Float64})
    env_data = env.py_env.step(a)
    env.info = env_data[end]
    env.rews = env_data[2]
    env.state = env_data[1]
    env.done = false
    env.t = Int(env_data[end]["elapsed_step"][1])
    return env
end

function _restore_using_acts!(env::EnvpoolEnv)
    env_data = nothing
    if isempty(env.acts)
        return env
    end
    for a in env.acts
        acts = repeat(a', env.num_envs)
        env_data = env.py_env.step(acts)
    end
    env.info = env_data[end]
    env.rews = env_data[2]
    env.state = env_data[1]
    env.done = false
    env.t = Int(env_data[end]["elapsed_step"][1])
    return env
end