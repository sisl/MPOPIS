
mutable struct MultiCarRacingEnv{A,T,R<:AbstractRNG} <: AbstractEnv
    N::Int
    envs::Vector{CarRacingEnv}
    action_space::A
    observation_space::Space{Vector{A}}
    state::Vector{T}
    done::Bool
    t::Int
    dt::T
    δt::T
    track::Track
    rng::R
end

"""
    MultiCarRacingEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `N = 2                                                     # Number of cars`   
- `dt = 0.1`,                                                # Time step between actions
- `δt = 0.01,                                                # Time step used for integration`
- `track = string(@__DIR__, "/car_racing_tracks/curve.csv"), # Track to load`
- 'car_params = []                                           # Vector of car parameters to use
- `rng = Random.GLOBAL_RNG`
"""
function MultiCarRacingEnv(N=2;
    T = Float64,
    dt = 0.1,
    δt = 0.01,
    track = string(@__DIR__, "/car_racing_tracks/curve.csv"),
    car_params = [],
    rng = Random.GLOBAL_RNG,
)
    
    length(car_params) <= N || error("# Car parameters must be ≤ # cars")

    envs = Vector{CarRacingEnv}(undef, N)
    for ii in 1:N
        if length(car_params) >= ii
            cre = CarRacingEnv(car_params, T=T, dt=dt, δt=δt, track=track, rng=rng)
        else
            cre = CarRacingEnv(T=T, dt=dt, δt=δt, track=track, rng=rng)
        end
        envs[ii] = cre
    end 

    
    endpts_l = []
    endpts_r = []
    single_state_size = length(state_space(envs[1]))
    obs_space_vec = Vector{ClosedInterval}(undef, N*single_state_size)
    state = zeros(T, N*single_state_size)
    for (idx, en) in enumerate(envs)
        endpts_l = [endpts_l; leftendpoint(action_space(en))]
        endpts_r = [endpts_r; rightendpoint(action_space(en))]
        start_idx = single_state_size*(idx-1) + 1
        end_idx = single_state_size*idx
        obs_space_vec[start_idx:end_idx] =  state_space(en)[:]
    end
    action_space = ClosedInterval{Vector{T}}(endpts_l, endpts_r)
    observation_space = Space(obs_space_vec)

    env = MultiCarRacingEnv(
        N,
        envs,
        action_space,
        observation_space,
        state,
        false,
        0,
        dt,
        δt,
        Track(track),
        rng,
    )

    reset!(env)
    env
end

MultiCarRacingEnv{T}(; kwargs...) where {T} = MultiCarRacingEnv(; T = T, kwargs...)

# Might not want to have the same seed for every envrionment here
function Random.seed!(env::MultiCarRacingEnv, seed)
    for en in env.envs
        Random.seed!(en.rng, seed)
    end
end

RLBase.action_space(env::MultiCarRacingEnv) = env.action_space
RLBase.state_space(env::MultiCarRacingEnv{T}) where {T} = env.observation_space
RLBase.is_terminated(env::MultiCarRacingEnv) = env.done
RLBase.state(env::MultiCarRacingEnv) = env.state

function _update_states_env2envs(env::MultiCarRacingEnv)
    for (idx, en) in enumerate(env.envs)
        ss_size = length(state_space(en))
        start_idx = ss_size*(idx-1) + 1
        end_idx = ss_size*idx
        en.state = env.state[start_idx:end_idx]
    end
end

function _update_states_envs2env(env::MultiCarRacingEnv)
    for (idx, en) in enumerate(env.envs)
        ss_size = length(state_space(en))
        start_idx = ss_size*(idx-1) + 1
        end_idx = ss_size*idx
        env.state[start_idx:end_idx] = en.state
    end
end

function within_track(env::MultiCarRacingEnv)
    within = true   
    for en in env.envs
        within = within && within_track(en).within
    end
    return within
end

function exceed_β(env::MultiCarRacingEnv)
    exceed = false
    for en in env.envs
        exceed = exceed || exceed_β(en)
    end
    return exceed
end


"""
    reward(env::MultiCarRacingEnv)
    - Sum over rewards of the ohter envrionments
    - Collision (≤ 4m): -7000
    - Penalize distance away by the distance amount
"""
function RLBase.reward(env::MultiCarRacingEnv{T}) where {T} 
    rew = 0.0
    for (ii, en) in enumerate(env.envs)
        rew += reward(en)
        for jj ∈ (ii+1):env.N
            Δd = norm(env.envs[jj].state[1:2] - en.state[1:2])
            rew += -Δd
            if Δd ≤ 4.0
                rew += -11000.0
            end
        end
    end
    return rew
end

function RLBase.reset!(env::MultiCarRacingEnv{A,T}) where {A,T}
    ss_size = length(env.state)
    ind_ss_size = round(Int, length(env.state)/env.N)
    env.envs[1].state = zeros(T, ind_ss_size)
    env.envs[1].state[3] = deg2rad(90)
    env.envs[1].state[4] = 10.0
    for ii ∈ 2:env.N
        if mod(ii,2) == 0 
            env.envs[ii].state[1] = ii/2*5.0
        else
            env.envs[ii].state[1] = (1-ii)/2*5.0
        end
        env.envs[ii].state[3] = deg2rad(90)
        env.envs[ii].state[4] = 10.0
    end

    _update_states_envs2env(env)
    env.t = 0
    env.done = false
    nothing
end

function RLBase.reset!(env::MultiCarRacingEnv{A,T}, state::Vector{T}) where {A,T}
    env.state = state
    _update_states_env2envs(env)
    env.t = 0
    env.done = false
    nothing
end

"""
    a::Vector{Float64}
    size = (2N,)
    a[1] = Turn angle [-max turn angle, max turn angle] (-1 right turn, +1 left turn)
    a[2] = Pedal amount (-1 = full brake, 1 = full throttle)
    a[3] = Turn angle Car 2
    a[4] = Pedal amount Car 2
    a[2N-1] = turn angle Car N
    a[2N] = Pedal amount Car N
"""
function (env::MultiCarRacingEnv)(a::Vector{Float64})
    length(a) == env.N * 2 || error("Action space of each car is of size 2")
    for (ii, en) in enumerate(env.envs)
        aᵢ = a[(2*ii-1):(2*ii)]
        _step!(en, aᵢ)
    end
    _update_states_envs2env(env)
end

function (env::MultiCarRacingEnv)(a::Vector{Int})
    env(Float64.(a))
end

function (env::MultiCarRacingEnv)(a::Matrix{Float64})
    size(a)[2] == 1 || error("Only implented for one step")
    env(vec(a))
end