
using ReinforcementLearning
using LinearAlgebra
using Distributions

include("../util/utils.jl")

struct MPPI_Policy_Params
    num_samples::Int        # Number of samples for the MPPI policy
    horizon::Int
    λ::Float64
    α::Float64
    U₀::Vector{Float64}
    ss::Int
    as::Int
    cs::Int
end

mutable struct MPPI_Policy{R<:AbstractRNG}
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
end

mutable struct GMPPI_Policy{R<:AbstractRNG}
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
end


function get_MPPI_policy_params(env::AbstractEnv; 
    type::Symbol,
    num_samples::Int = 50, 
    horizon::Int = 50, 
    λ::Float64 = 1.0, 
    α::Float64 = 1.0, 
    U₀::Vector{Float64} = [0.0],
    cov_mat::Union{Matrix{Float64},Vector{Float64}} = [1.0],
    rng::AbstractRNG = Random.GLOBAL_RNG,
    )
    
    ss = length(env.state)                    # State space size
    as = action_space_size(action_space(env)) # Action space size
    cs = as * horizon                         # Control size (number of actions per sample)

    if length(U₀) == as
        U₀ = repeat(U₀, horizon)
    end
    length(U₀) == cs || error("U₀ must be length of action space or control space")

    if type == :mppi
        repeat_num = 1
        check_size = as
    elseif type == :gmppi
        repeat_num = horizon
        check_size = cs
    else
        error("Incorrect type for MPPPI")
    end

    if size(cov_mat)[1] == as
        cov_mat = block_diagm(cov_mat, repeat_num)
    end
    size(cov_mat)[1] == check_size || error("Covariance matrix size problem")
    size(cov_mat)[1] == size(cov_mat)[2] || error("Covriance must be square")
    Σ = cov_mat

    params = MPPI_Policy_Params(num_samples, horizon, λ, α, U₀, ss, as, cs)
    return params, U₀, Σ, rng
end

function MPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng = get_MPPI_policy_params(env; type=:mppi, kwargs...)
    return MPPI_Policy(params, env, U₀, Σ, rng)
end

function GMPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng = get_MPPI_policy_params(env; type=:gmppi, kwargs...)
    return GMPPI_Policy(params, env, U₀, Σ, rng)
end

Random.seed!(pol::MPPI_Policy, seed) = Random.seed!(pol.rng, seed)
Random.seed!(pol::GMPPI_Policy, seed) = Random.seed!(pol.rng, seed)

function (pol::MPPI_Policy)(env::AbstractEnv)
    K = pol.params.num_samples
    T = pol.params.horizon
    as = pol.params.as
    cs = pol.params.cs
    λ = pol.params.λ
    α = pol.params.α
    γ = λ*(1-α)
    
    # Get samples for which our trajectories will be defined
    E = rand(pol.rng, Distributions.MvNormal(pol.Σ), K, T)
    Σ_inv = inv(pol.Σ)
    trajectory_cost = zeros(Float64, K)

    # Threads.@threads for ii = 1:num_samples
    for k ∈ 1:K
        sim_env = copy(env) # Slower, but allows for multi threading

        for t ∈ 1:T
            uₜ = pol.U[((t-1)*as+1):(t*as)]
            Vₜ = uₜ + E[k,t]
            control_cost = γ * uₜ'*Σ_inv*E[k,t]  

            model_controls = get_model_controls(action_space(env), Vₜ)
            if as == 1
                sim_env(model_controls[1])
            else
                sim_env(model_controls)
            end
            # Subtrating based on "reward", Addingg based on "cost"
            trajectory_cost[k] = trajectory_cost[k] - reward(sim_env) + control_cost
        end 
    end
    
    # Information-Theoretic Weight Computation (soft max)
    weights = compute_weights(trajectory_cost, λ) 
    weights = reshape(weights, K, 1)

    # Weight the noise based on the IT weights
    weighted_noise = zeros(Float64, cs)
    for t ∈ 1:T
        for k ∈ 1:K
            weighted_noise[((t-1)*as+1):(t*as)] += weights[k] .* E[k,t]
        end
    end
    weighted_controls = pol.U + weighted_noise

    # Get control (action set for the first time step)
    control = get_model_controls(action_space(env), weighted_controls[1:as])

    # Roll the control policy so next interation we start with a mean of pol.U
    pol.U[1:(end-as)] = weighted_controls[(as+1):end]
    pol.U[(end-as):end] = pol.params.U₀[(end-as):end]
    
    return control
end

function (pol::GMPPI_Policy)(env::AbstractEnv)
    K = pol.params.num_samples
    T = pol.params.horizon
    as = pol.params.as
    cs = pol.params.cs
    λ = pol.params.λ
    α = pol.params.α
    γ = λ*(1-α)
    
    # Get samples for which our trajectories will be defined
    E = rand(pol.rng, Distributions.MvNormal(pol.Σ), K)
    Σ_inv = inv(pol.Σ)
    trajectory_cost = zeros(Float64, K)

    # Threads.@threads for ii = 1:num_samples
    for k ∈ 1:K
        sim_env = copy(env) # Slower, but allows for multi threading
        
        Vₖ = pol.U + E[:,k]
        control_cost = γ * pol.U'*Σ_inv*E[:,k]

        model_controls = get_model_controls(action_space(env), Vₖ, T)
        for t ∈ 1:T
            if as == 1
                sim_env(model_controls[t])
            else
                sim_env(model_controls[:,t])
            end
            trajectory_cost[k] -= reward(sim_env) # Subtrating based on "reward"
        end
        trajectory_cost[k] += control_cost  # Addingg based on "cost"
    end
    
    # Information-Theoretic Weight Computation (soft max)
    weights = compute_weights(trajectory_cost, λ) 
    
    # Weight the noise based on the IT weights
    weighted_noise = zeros(Float64, cs)
    for rᵢ ∈ 1:cs
        weighted_noise[rᵢ] = weights' * E[rᵢ,:]
    end
    weighted_controls = pol.U + weighted_noise

    # Get control (action set for the first time step)
    control = get_model_controls(action_space(env), weighted_controls[1:as])

    # Roll the control policy so next interation we start with a mean of pol.U
    pol.U[1:(end-as)] = weighted_controls[(as+1):end]
    pol.U[(end-as):end] = pol.params.U₀[(end-as):end]
    
    return control
end