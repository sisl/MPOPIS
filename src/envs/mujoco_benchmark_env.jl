
using PyCall

mutable struct MuJoCoBenchmarkEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    domain_name::String
    task::String
    py_env
    env_data
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    observation_keys::Base.KeySet
    num_states::Int
    done::Bool
    t::Int
    dt::T
    δt::T
    rng::R
end

"""
    MuJoCoBenchmarkEnv(;kwargs...)

# Keyword arguments

"""
function MuJoCoBenchmarkEnv(
    domain_name = "cartpole",
    task = "balance";
    T = Float64,
    dt = 0.1,
    δt = 0.01,
    random_seed = 42,
    rng = Random.GLOBAL_RNG,
)

    py"""
    from dm_control import mujoco
    from dm_control import suite
    import numpy as np
    def get_env(domain_name, task, random_seed=42):
        random_state = np.random.RandomState(random_seed)
        return suite.load(domain_name, task, task_kwargs={'random': random_state})
    """

    py_env = py"get_env"(domain_name, task, random_seed)
    py_env.physics.model.opt.timestep = δt
    env_data = py_env.reset()
    action_spec = py_env.action_spec()

    action_space = ClosedInterval{Vector{T}}(
        action_spec.minimum,
        action_spec.maximum,
        )

    num_of_obs = 0
    observation_spec = py_env.observation_spec()
    observation_keys = keys(observation_spec)
    for key in observation_keys
        num_of_obs += observation_spec[key].shape[1]
    end
    observation_vec = [-Inf..Inf for _ in 1:num_of_obs]
    observation_space = Space(observation_vec)

    env = MuJoCoBenchmarkEnv(
        domain_name,
        task,
        py_env,
        env_data,
        action_space,
        observation_space,
        observation_keys,
        num_of_obs,
        false,
        0,
        dt,
        δt,
        rng,
    )

    reset!(env)
    env
end

# CarRacingEnv{T}(; kwargs...) where {T} = CarRacingEnv(; T = T, kwargs...)

Random.seed!(env::MuJoCoBenchmarkEnv, seed) = Random.seed!(env.rng, seed)

RLBase.action_space(env::MuJoCoBenchmarkEnv) = env.action_space
RLBase.state_space(env::MuJoCoBenchmarkEnv{T}) where {T} = env.observation_space
RLBase.is_terminated(env::MuJoCoBenchmarkEnv) = env.done

function RLBase.state(env::MuJoCoBenchmarkEnv{A, T, R}) where {A, T, R}
    states = zeros(T, env.num_states)
    indx = 1
    for key in env.observation_keys
        len_of_obs = env.py_env.observation_spec()[key].shape[1]
        states[indx:(len_of_obs + indx - 1)] = env.env_data[end][key]
        indx += len_of_obs
    end
    return states
end

"""
    reward(env::CarRacingEnv)

    - Reward for going fast, penalty for boundary violation,
        excess β and distance from center of track
    - Boundary violation    : -1000000
    - Excessive β           :  -5000
    - Distance from center  : -(x² + y²)^(1/2)
    - Speed                 : +2.0 * (Vx² + Vy²)^(1/2)
"""
function RLBase.reward(env::MuJoCoBenchmarkEnv{T}) where {T}
    rew = env.env_data[2]
    if isnothing(rew)
        return 0.0
    else
        return rew
    end
end

function RLBase.reset!(env::MuJoCoBenchmarkEnv{A,T}) where {A,T}
    env_data = env.py_env.reset()



    env.t = 0
    env.done = false
    return nothing
end

# function RLBase.reset!(env::MuJoCoBenchmarkEnv{A,T}, state::Vector{T}) where {A,T}
#     env.state = state
#     env.t = 0
#     env.done = false
#     return nothing
# end

"""
    a::Vector{Float64}
"""
function (env::MuJoCoBenchmarkEnv{<:ClosedInterval})(a::Vector{Float64})
    a in env.action_space || error("Action is not in action space")
    _step!(env, a)
end

function (env::MuJoCoBenchmarkEnv{<:ClosedInterval})(a::Vector{Int})
    env(Float64.(a))
end

function (env::MuJoCoBenchmarkEnv{<:ClosedInterval})(a::Matrix{Float64})
    size(a)[2] == 1 || error("Only implented for one step")
    env(vec(a))
end

"""

"""
function _step!(env::MuJoCoBenchmarkEnv, a::Vector{Float64})
    env.t += 1
    integration_steps = round(Int, env.dt / env.δt)
    for _ in 1:integration_steps
        env.env_data = env.py_env.step(a)
    end
    return env
end
