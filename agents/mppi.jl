
using ReinforcementLearning
using LinearAlgebra
using Distributions

using CovarianceEstimation
import CovarianceEstimation.LinearShrinkageTarget

include("../util/utils.jl")

mutable struct MPPI_Logger
    trajectories::Vector{Matrix{Float64}}
    traj_costs::Vector{Float64}
    traj_weights::Vector{Float64}
end

struct MPPI_Policy_Params{M<:AbstractWeightMethod}
    num_samples::Int        # Number of samples for the MPPI policy
    horizon::Int
    λ::Float64
    α::Float64
    U₀::Vector{Float64}
    ss::Int
    as::Int
    cs::Int
    weight_method::M
    min_max_sample_threshold::Int
    log::Bool
end

mutable struct MPPI_Policy{R<:AbstractRNG} <: AbstractPathIntegralPolicy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
    logger::MPPI_Logger
end

mutable struct GMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
    logger::MPPI_Logger
end

mutable struct CEMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    ce_elite_threshold::Float64
    Σ_estimation_method::LinearShrinkage
    rng::R
    logger::MPPI_Logger
end

mutable struct CMAMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    σ::Float64
    m_elite::Int
    ws::Vector{Float64}
    μ_eff::Float64
    cσ::Float64
    dσ::Float64
    cΣ::Float64
    c1::Float64
    cμ::Float64
    E::Float64
    rng::R
    logger::MPPI_Logger
end

mutable struct NESMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    A::Matrix{Float64}
    opt_its::Int
    step_factor::Float64
    rng::R
    logger::MPPI_Logger
end

function get_MPPI_policy_params(env::AbstractEnv, type::Symbol;
    num_samples::Int = 50, 
    horizon::Int = 50, 
    λ::Float64 = 1.0, 
    α::Float64 = 1.0, 
    U₀::Vector{Float64} = [0.0],
    cov_mat::Union{Matrix{Float64},Vector{Float64}} = [1.0],
    weight_method::Symbol = :IT,
    elite_threshold::Float64 = 0.8,
    min_max_sample_perc::Float64 = 0.0,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    log::Bool = false,
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

    if weight_method == :IT
        weight_m = Information_Theoretic(λ)
    elseif cost_method == :CE
        n = round(Int, num_samples*(1-elite_threshold))
        weight_m = Cross_Entropy(elite_threshold, n)
    else
        error("No cost method implemented for $weight_method")
    end

    min_max_sample_threshold = round(Int, num_samples*min_max_sample_perc)
    
    log_traj = [Matrix{Float64}(undef, (horizon, ss)) for _ in 1:num_samples]
    log_traj_costs = Vector{Float64}(undef, num_samples)
    log_traj_weights = Vector{Float64}(undef, num_samples)
    mppi_logger = MPPI_Logger(log_traj, log_traj_costs, log_traj_weights)

    params = MPPI_Policy_Params(
        num_samples, horizon, λ, α, U₀, ss, as, cs, 
        weight_m, min_max_sample_threshold, log
    )
    return params, U₀, Σ, rng, mppi_logger
end

function MPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng, mppi_logger = get_MPPI_policy_params(env, :mppi; kwargs...)
    return MPPI_Policy(params, env, U₀, Σ, rng, mppi_logger)
end

function GMPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng, mppi_logger = get_MPPI_policy_params(env, :gmppi; kwargs...)
    return GMPPI_Policy(params, env, U₀, Σ, rng, mppi_logger)
end

"""
https://mateuszbaran.github.io/CovarianceEstimation.jl/dev/man/lshrink/
    Covariance Estimation shrinkage estimators:
        - :lw = Lediot & Wolf (http://www.ledoit.net/honey.pdf)
        - :ss = Schaffer & Strimmer (https://strimmerlab.github.io/)
        - :rblw = Rao-Blackwellised estimator (https://arxiv.org/pdf/0907.4698.pdf)
        - :oas = Oracle-Approximating (https://arxiv.org/pdf/0907.4698.pdf)
    Covariance Estimation targets
        - DiagonalUnitVariance
        - DiagonalCommonVariance
        - DiagonalUnequalVariance
        - CommonCovariance
        - PerfectPositiveCorrelation
        - ConstantCorrelation
"""
function CEMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    ce_elite_threshold::Float64 = 0.8,
    Σ_est_target::LinearShrinkageTarget = DiagonalUnequalVariance(),
    Σ_est_shrinkage::Symbol = :lw,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = get_MPPI_policy_params(env, :gmppi; kwargs...)
    Σ_est_method = LinearShrinkage(Σ_est_target, Σ_est_shrinkage)
    pol = CEMPPI_Policy(params, env, U₀, 
              Σ, opt_its, ce_elite_threshold, 
              Σ_est_method, rng, mppi_logger,
          )
    return pol
end

function CMAMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    σ::Float64 = 1.0,
    elite_perc_threshold::Float64 = 0.8,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = get_MPPI_policy_params(env, :gmppi; kwargs...)
    m = params.num_samples
    n = params.cs
    m_elite = round(Int, (1.0-elite_perc_threshold)*m)
    ws = log((m+1)/2) .- log.(1:m)
    ws[1:m_elite] ./= sum(ws[1:m_elite])
    μ_eff = 1 / sum(ws[1:m_elite].^2)
    cσ = (μ_eff + 2)/(n + μ_eff + 5)
    dσ = 1 + 2max(0, sqrt((μ_eff-1)/(n+1))-1) + cσ
    cΣ = (4 + μ_eff/n)/(n + 4 + 2μ_eff/n)
    c1 = 2/((n+1.3)^2 + μ_eff)
    cμ = min(1-c1, 2*(μ_eff-2+1/μ_eff)/((n+2)^2 + μ_eff))
    ws[m_elite+1:end] .*= -(1 + c1/cμ)/sum(ws[m_elite+1:end])
    E = n^0.5*(1-1/(4n)+1/(21*n^2))

    pol = CMAMPPI_Policy(params, env, U₀, Σ, opt_its, σ, m_elite,
        ws, μ_eff, cσ, dσ, cΣ, c1, cμ, E, rng, mppi_logger)
    return pol
end

function NESMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    step_factor::Float64 = 0.01,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = get_MPPI_policy_params(env, :gmppi; kwargs...)
    A = sqrt(Σ)
    pol = NESMPPI_Policy(params, env, U₀, Σ, A, opt_its, step_factor, rng, mppi_logger)
    return pol
end

Random.seed!(pol::AbstractPathIntegralPolicy, seed) = Random.seed!(pol.rng, seed)

function (pol::MPPI_Policy)(env::AbstractEnv)
    K, T = pol.params.num_samples, pol.params.horizon
    as, cs = pol.params.as, pol.params.cs
    trajectory_cost, E = calculate_trajectory_costs(pol, env)
    
    # Compute weights based on weight method
    weights = compute_weights(pol.params.weight_method, trajectory_cost) 
    weights = reshape(weights, K, 1)

    # Weight the noise based on the calcualted weights
    weighted_noise = zeros(Float64, cs)
    for t ∈ 1:T
        for k ∈ 1:K
            weighted_noise[((t-1)*as+1):(t*as)] += weights[k] .* E[k,t]
        end
    end
    weighted_controls = pol.U + weighted_noise
    control = get_controls_roll_U!(pol, weighted_controls)

    if pol.params.log
        pol.logger.traj_costs = trajectory_cost
        pol.logger.traj_weights = vec(weights)
    end

    return control
end

function (pol::AbstractGMPPI_Policy)(env::AbstractEnv)
    cs = pol.params.cs
    trajectory_cost, E = calculate_trajectory_costs(pol, env)
    
    # Compute weights based on weight method
    weights = compute_weights(pol.params.weight_method, trajectory_cost) 
    
    # Weight the noise based on the calcualted weights
    weighted_noise = zeros(Float64, cs)
    for rᵢ ∈ 1:cs
        weighted_noise[rᵢ] = weights' * E[rᵢ,:]
    end
    weighted_controls = pol.U + weighted_noise
    control = get_controls_roll_U!(pol, weighted_controls)

    if pol.params.log
        pol.logger.traj_costs = trajectory_cost
        pol.logger.traj_weights = weights
    end
    return control
end

function get_controls_roll_U!(pol::AbstractPathIntegralPolicy, weighted_controls::Vector)
    as = pol.params.as
    # Get control (action set for the first time step)
    control = get_model_controls(action_space(pol.env), weighted_controls[1:as])

    # Roll the control policy so next interation we start with a mean of pol.U
    if pol.params.horizon > 1
        pol.U[1:(end-as)] = weighted_controls[(as+1):end]
        pol.U[(end-as):end] = pol.params.U₀[(end-as):end]
    else
        pol.U = weighted_controls
    end
    return control
end

function calculate_trajectory_costs(pol::MPPI_Policy, env::AbstractEnv)
    K, T = pol.params.num_samples, pol.params.horizon
    as = pol.params.as
    γ = pol.params.λ*(1-pol.params.α)

    # Get samples for which our trajectories will be defined
    P = Distributions.MvNormal(pol.Σ)
    E = rand(pol.rng, P, K, T)
    Σ_inv = Distributions.invcov(P)
    
    # Get a few "extreme" samples for our distribution
    if pol.params.min_max_sample_threshold > 0
        ΔE = get_minmax_control_Δ(pol, env)
        num_minmax_u = size(ΔE, 2)
        num_minmax_samples = min(num_minmax_u, pol.params.min_max_sample_threshold)

        if num_minmax_u > pol.params.min_max_sample_threshold
            sample_idxs = sample(2:(num_minmax_u-1), pol.params.min_max_sample_threshold-3)
            # Add in the all max, all min, and mean controls
            sample_idxs = vcat(sample_idxs, [1, num_minmax_u-1, num_minmax_u])
        else 
            sample_idxs = 1:num_minmax_u
        end
        ΔE = ΔE[:, sample_idxs]
    end
    
    trajectory_cost = zeros(Float64, K)
    Threads.@threads for k ∈ 1:K
        sim_env = copy(env) # Slower, but allows for multi threading
        for t ∈ 1:T
            Eᵢ = E[k,t]
            if pol.params.min_max_sample_threshold > 0 && k <= num_minmax_samples
                Eᵢ = ΔE[((t-1)*as+1):(t*as), k]
            end
            uₜ = pol.U[((t-1)*as+1):(t*as)]
            Vₜ = uₜ + Eᵢ
            control_cost = γ * uₜ'*Σ_inv*Eᵢ
            model_controls = get_model_controls(action_space(sim_env), Vₜ)
            sim_env(model_controls)
            # Subtrating based on "reward", Adding based on "cost"
            trajectory_cost[k] = trajectory_cost[k] - reward(sim_env) + control_cost
            if pol.params.log
                pol.logger.trajectories[k][t, :] = sim_env.state
            end
        end
    end
    return trajectory_cost, E
end

function calculate_trajectory_costs(pol::GMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples

    # Get samples for which our trajectories will be defined
    P = Distributions.MvNormal(pol.Σ)
    E = rand(pol.rng, P, K)
    Σ_inv = Distributions.invcov(P)

    if pol.params.min_max_sample_threshold > 0
        ΔE = get_minmax_control_Δ(pol, env)
        num_minmax_u = size(ΔE, 2)
        num_minmax_samples = min(num_minmax_u, pol.params.min_max_sample_threshold)
        if num_minmax_u > pol.params.min_max_sample_threshold
            sample_idxs = sample(2:(num_minmax_u-1), pol.params.min_max_sample_threshold-3)
            # Add in the all max, all min, and mean controls
            sample_idxs = vcat(sample_idxs, [1, num_minmax_u-1, num_minmax_u])
        else 
            sample_idxs = 1:num_minmax_u
        end
        E[:, 1:num_minmax_samples] = ΔE[:, sample_idxs]
    end

    # Use the samples to simulate our model to get the costs
    trajectory_cost = simulate_model(pol, env, E, Σ_inv, pol.U)

    return trajectory_cost, E
end

function calculate_trajectory_costs(pol::CEMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    m_elite = round(Int, K*(1-pol.ce_elite_threshold))

    # Initial covariance of distribution
    U_orig = pol.U
    Σ′ = pol.Σ

    trajectory_cost = zeros(Float64, K)
    best_traj_cost = Inf
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Get a few "extreme" samples for our distribution
        if n == 1 && pol.params.min_max_sample_threshold > 0
            ΔE = get_minmax_control_Δ(pol, env)
            num_minmax_u = size(ΔE, 2)
            num_minmax_samples = min(num_minmax_u, pol.params.min_max_sample_threshold)
            if num_minmax_u > pol.params.min_max_sample_threshold
                sample_idxs = sample(2:(num_minmax_u-1), pol.params.min_max_sample_threshold-3)
                # Add in the all max, all min, and mean controls
                sample_idxs = vcat(sample_idxs, [1, num_minmax_u-1, num_minmax_u])
            else 
                sample_idxs = 1:num_minmax_u
            end
            E[:, 1:num_minmax_samples] = ΔE[:, sample_idxs]
        end

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            # Select elite samples, fit new distribution
            order = sortperm(trajectory_cost)
            elite = E[:, order[1:m_elite]]
            
            elite_traj_cost = trajectory_cost[order[1:m_elite]]
            if maximum(abs.(diff(elite_traj_cost, dims=1))) < 10e-3
                break
            end

            # Transposing elite based on needed format (n x p)
            Σ′ = cov(pol.Σ_estimation_method, elite') + 10e-9*I
            pol.U = pol.U + vec(mean(elite, dims=2))
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E
end

function calculate_trajectory_costs(pol::CMAMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    cs = pol.params.cs
    σ = pol.σ
    m_elite = pol.m_elite
    ws = pol.ws
    μ_eff = pol.μ_eff
    cσ = pol.cσ
    dσ = pol.dσ
    cΣ = pol.cΣ
    c1 = pol.c1
    cμ = pol.cμ
    E_cma = pol.E

    # Initial covariance of distribution
    U_orig = pol.U
    Σ = pol.Σ

    pσ, pΣ = zeros(pol.params.cs), zeros(pol.params.cs)
    trajectory_cost = zeros(Float64, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(σ^2*Σ)
        Σ_inv = Distributions.invcov(P)
        E = rand(pol.rng, P, K)

        # Get a few "extreme" samples for our distribution
        if n == 1 && pol.params.min_max_sample_threshold > 0
            ΔE = get_minmax_control_Δ(pol, env)
            num_minmax_u = size(ΔE, 2)
            num_minmax_samples = min(num_minmax_u, pol.params.min_max_sample_threshold)
            if num_minmax_u > pol.params.min_max_sample_threshold
                sample_idxs = sample(2:(num_minmax_u-1), pol.params.min_max_sample_threshold-3)
                # Add in the all max, all min, and mean controls
                sample_idxs = vcat(sample_idxs, [1, num_minmax_u-1, num_minmax_u])
            else 
                sample_idxs = 1:num_minmax_u
            end
            E[:, 1:num_minmax_samples] = ΔE[:, sample_idxs]
        end

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)

        if n < N
            # Reorder, select elite samples, fit new distribution
            order = sortperm(trajectory_cost)
            elite_E = E[:, order[1:m_elite]]

            elite_traj_cost = trajectory_cost[order[1:m_elite]]
            if maximum(abs.(diff(elite_traj_cost, dims=1))) < 10e-3
                break
            end

            # selection and mean update
            δs = elite_E/σ 
            δw = zeros(Float64, cs)
            for rᵢ ∈ 1:cs
                δw[rᵢ] = ws[1:m_elite]' * elite_E[rᵢ,:]
            end
            pol.U += σ*δw
            
            # step-size control
            C = Σ^-0.5
            pσ = (1-cσ)*pσ + sqrt(cσ*(2-cσ)*μ_eff)*C*δw
            σ *= exp(cσ/dσ * (norm(pσ)/E_cma - 1))

            # covariance adaptation
            hσ = Int(norm(pσ)/sqrt(1-(1-cσ)^(2n)) < (1.4+2/(cs+1))*E_cma)
            pΣ = (1-cΣ)*pΣ + hσ*sqrt(cΣ*(2-cΣ)*μ_eff)*δw

            temp_sum = 0
            for ii in 1:K
                if ws[ii] ≥ 0
                    w0 = ws[ii]
                else
                    w0 = n*ws[ii]/norm(C*δs[order[ii]])^2
                end
                temp_sum += w0*δs[order[ii]]*δs[order[ii]]'
            end

            Σ = (1-c1-cμ)*Σ + c1*(pΣ*pΣ' + (1-hσ)*cΣ*(2-cΣ)*Σ) .+ cμ*temp_sum
            Σ = triu(Σ)+triu(Σ,1)' # enforce symmetry
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E
end

function calculate_trajectory_costs(pol::NESMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its

    # Initial covariance and principal matrix
    U_orig = pol.U
    Σ′ = pol.Σ
    A′ = pol.A

    trajectory_cost = zeros(Float64, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if maximum(abs.(diff(trajectory_cost, dims=1))) < 10e-3
            break
        end

        if n < N
            ∇μlog_p_x = zeros(Float64, size(pol.U))
            ∇Alog_p_x = zeros(Float64, size(pol.Σ))
            for k ∈ 1:K
                ∇μlog_p_x .+= Σ_inv * E[:,k] .* trajectory_cost[k]
                ∇Σlog_p_x = 1/2*Σ_inv*E[:,k]*E[:,k]'*Σ_inv - 1/2*Σ_inv
                ∇Alog_p_x += A′*(∇Σlog_p_x + ∇Σlog_p_x') * trajectory_cost[k]
            end
            A′ -= pol.step_factor/K .* ∇Alog_p_x ./ K
            Σ′ = A′' * A′
            pol.U -= pol.step_factor/K .* ∇μlog_p_x
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E
end

function get_minmax_control_Δ(pol::AbstractPathIntegralPolicy, env::AbstractEnv)
    min_max_controls = get_min_max_control_set(action_space(env), pol.params.horizon)
    # Matrix size of (cs x number of augmented sample size)
    ΔU = Matrix{Float64}(undef, pol.params.cs, length(min_max_controls))
    
    for ii ∈ 1:length(min_max_controls)
        uᵢ = vec(reshape(min_max_controls[ii], :, 1))
        ΔU[:, ii] = uᵢ - pol.U
    end
    return ΔU
end

function simulate_model(pol::AbstractGMPPI_Policy, env::AbstractEnv, 
    E::Matrix{Float64}, Σ_inv::Matrix{Float64}, U_orig::Vector{Float64},
)
    K, T = pol.params.num_samples, pol.params.horizon
    as = pol.params.as
    γ = pol.params.λ*(1-pol.params.α)

    trajectory_cost = zeros(Float64, K)
    Threads.@threads for k ∈ 1:K
        sim_env = copy(env) # Slower, but allows for multi threading
        Vₖ = pol.U + E[:,k]
        control_cost = γ * U_orig'*Σ_inv*(Vₖ .- U_orig) 
        model_controls = get_model_controls(action_space(sim_env), Vₖ, T)
        trajectory_cost[k] = rollout_model(sim_env, T, model_controls, pol, k)
        trajectory_cost[k] += control_cost  # Adding based on "cost"
    end
    return trajectory_cost
end

function rollout_model(env::AbstractEnv, T::Int, model_controls::Vector, 
    pol::AbstractPathIntegralPolicy, k::Int)
    model_controls_mat = reshape(model_controls, size(model_controls, 1), 1)
    rollout_model(env, T, model_controls_mat, pol, k)
end

function rollout_model(env::AbstractEnv, T::Int, model_controls::Matrix, 
    pol::AbstractPathIntegralPolicy, k::Int,
)
    as = pol.params.as
    traj_cost = 0.0
    for t ∈ 1:T
        controls = as == 1 ? model_controls[t] : model_controls[:,t]
        env(controls)
        traj_cost -= reward(env) # Subtracting based on "reward"
        if pol.params.log
            pol.logger.trajectories[k][t, :] = env.state
        end
    end
    return traj_cost
end