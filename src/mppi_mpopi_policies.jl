
mutable struct MPPI_Logger
    trajectories::Vector{Matrix{Float64}}
    traj_costs::Vector{Float64}
    traj_weights::Vector{Float64}
end

struct MPPI_Policy_Params{M<:AbstractWeightMethod}
    num_samples::Int      
    horizon::Int
    λ::Float64
    α::Float64
    U₀::Vector{Float64}
    ss::Int
    as::Int
    cs::Int
    weight_method::M
    log::Bool
end

""" 
MPPI_Policy_Params(env::AbstractEnv, type::Symbol; kwargs...)
    Construct the mppi policy parameter struct
kwargs:
    - num_samples::Int = 50, 
    - horizon::Int = 50, 
    - λ::Float64 = 1.0, 
    - α::Float64 = 1.0, 
    - U₀::Vector{Float64} = [0.0],
    - cov_mat::Union{Matrix{Float64},Vector{Float64}} = [1.0],
    - weight_method::Symbol = :IT,
    - elite_threshold::Float64 = 0.8,
    - rng::AbstractRNG = Random.GLOBAL_RNG,
    - log::Bool = false,    
"""
function MPPI_Policy_Params(env::AbstractEnv, type::Symbol;
    num_samples::Int = 50, 
    horizon::Int = 50, 
    λ::Float64 = 1.0, 
    α::Float64 = 1.0, 
    U₀::Vector{Float64} = [0.0],
    cov_mat::Union{Matrix{Float64},Vector{Float64}} = [1.0],
    weight_method::Symbol = :IT,
    elite_threshold::Float64 = 0.8,
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
    
    log_traj = [Matrix{Float64}(undef, (horizon, ss)) for _ in 1:num_samples]
    log_traj_costs = Vector{Float64}(undef, num_samples)
    log_traj_weights = Vector{Float64}(undef, num_samples)
    mppi_logger = MPPI_Logger(log_traj, log_traj_costs, log_traj_weights)

    params = MPPI_Policy_Params(
        num_samples, horizon, λ, α, U₀, ss, as, cs, 
        weight_m, log
    )
    return params, U₀, Σ, rng, mppi_logger
end

#######################################
# MPPI
#######################################
mutable struct MPPI_Policy{R<:AbstractRNG} <: AbstractPathIntegralPolicy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
    logger::MPPI_Logger
end

function MPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :mppi; kwargs...)
    return MPPI_Policy(params, env, U₀, Σ, rng, mppi_logger)
end

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

function calculate_trajectory_costs(pol::MPPI_Policy, env::AbstractEnv)
    K, T = pol.params.num_samples, pol.params.horizon
    as = pol.params.as
    γ = pol.params.λ*(1-pol.params.α)

    # Get samples for which our trajectories will be defined
    P = Distributions.MvNormal(pol.Σ)
    E = rand(pol.rng, P, K, T)
    Σ_inv = Distributions.invcov(P)
    
    trajectory_cost = zeros(Float64, K)
    Threads.@threads for k ∈ 1:K
        sim_env = copy(env) # Slower, but allows for multi threading
        for t ∈ 1:T
            Eᵢ = E[k,t]
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

#######################################
# GMPPI Policies
#######################################
function (pol::AbstractGMPPI_Policy)(env::AbstractEnv)
    cs = pol.params.cs
    trajectory_cost, E, weights = calculate_trajectory_costs(pol, env) 
    
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

function simulate_model(pol::AbstractGMPPI_Policy, env::AbstractEnv, 
    E::Matrix{Float64}, Σ_inv::Matrix{Float64}, U_orig::Vector{Float64},
    n::Int=1,
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
        trajectory_cost[k] = rollout_model(sim_env, T, model_controls, pol, k, n)
        trajectory_cost[k] += control_cost  # Adding based on "cost"
    end
    return trajectory_cost
end

"""
GMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Generalized MPPI policy strcut
"""
mutable struct GMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    rng::R
    logger::MPPI_Logger
end

"""
GMPPI_Policy(env::AbstractEnv; kwargs...)
    - env::AbstractEnv    
kwargs are passed to MPPI_Policy_params
"""
function GMPPI_Policy(env::AbstractEnv; kwargs...)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return GMPPI_Policy(params, env, U₀, Σ, rng, mppi_logger)
end

function calculate_trajectory_costs(pol::GMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples

    # Get samples for which our trajectories will be defined
    P = Distributions.MvNormal(pol.Σ)
    E = rand(pol.rng, P, K)
    Σ_inv = Distributions.invcov(P)

    # Use the samples to simulate our model to get the costs
    trajectory_cost = simulate_model(pol, env, E, Σ_inv, pol.U)
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end

"""
CEMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Cross-Entropy version of MPOPI
"""
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

"""
CEMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    ce_elite_threshold::Float64 = 0.8,
    Σ_est::Symbol = :mle,
    kwargs...
kwargs passed to MPPI_Policy_Params

Options for Σ_est
    - :mle = maximum liklihood estimation
    - :lw = Lediot & Wolf (http://www.ledoit.net/honey.pdf)
    - :ss = Schaffer & Strimmer (https://strimmerlab.github.io/)
    - :rblw = Rao-Blackwell estimator (https://arxiv.org/pdf/0907.4698.pdf)
    - :oas = Oracle-Approximating (https://arxiv.org/pdf/0907.4698.pdf)
https://mateuszbaran.github.io/CovarianceEstimation.jl/dev/man/lshrink/
"""
function CEMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    ce_elite_threshold::Float64 = 0.8,
    Σ_est::Symbol = :mle,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    if Σ_est == :mle
        Σ_est_method = SimpleCovariance()
    elseif  Σ_est == :lw
        Σ_est_method = LinearShrinkage(DiagonalUnequalVariance(), :lw)
    elseif Σ_est == :ss
        Σ_est_method = LinearShrinkage(DiagonalUnequalVariance(), :ss)
    elseif Σ_est == :rblw
        Σ_est_method = LinearShrinkage(DiagonalCommonVariance(), :rblw)
    elseif Σ_est == :oas
        Σ_est_method = LinearShrinkage(DiagonalCommonVariance(), :oas)
    else
        error("CEMPPI_Policy - Not a valid Σ estimation method")
    end    
    pol = CEMPPI_Policy(params, env, U₀, 
              Σ, opt_its, ce_elite_threshold, 
              Σ_est_method, rng, mppi_logger,
          )
    return pol
end

function calculate_trajectory_costs(pol::CEMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    m_elite = round(Int, K*(1-pol.ce_elite_threshold))

    # Initial covariance of distribution
    U_orig = pol.U
    Σ′ = pol.Σ

    trajectory_cost = zeros(Float64, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

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

            # (μ′, Σ′) = StatsBase.mean_and_cov(elite, 2)
            # Σ′ = Σ′ + + 10e-9*I
            # pol.U = pol.U + vec(μ′)

            # Transposing elite based on needed format (n x p)
            Σ′ = cov(pol.Σ_estimation_method, elite') + 10e-9*I
            pol.U = pol.U + vec(mean(elite, dims=2))
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end

"""
CMAMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Covariance Matrix Adaptation version of MPOPI
"""
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

"""
CMAMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    σ::Float64 = 1.0,
    elite_perc_threshold::Float64 = 0.8,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function CMAMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    σ::Float64 = 1.0,
    elite_perc_threshold::Float64 = 0.8,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
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

function calculate_trajectory_costs(pol::CMAMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    cs = pol.params.cs
    σ = pol.σ
    m_elite = pol.m_elite
    ws, μ_eff, cσ, dσ = pol.ws, pol.μ_eff, pol.cσ, pol.dσ
    cΣ, c1, cμ, E_cma = pol.cΣ, pol.c1, pol.cμ, pol.E

    # Initial covariance of distribution
    U_orig = pol.U
    Σ = pol.Σ

    pσ, pΣ = zeros(pol.params.cs), zeros(pol.params.cs)
    trajectory_cost = zeros(Float64, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        if N > 1
            P = Distributions.MvNormal(σ^2*Σ)
        else
            P = Distributions.MvNormal(Σ)
        end
        Σ_inv = Distributions.invcov(P)
        E = rand(pol.rng, P, K)

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
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end

"""
μAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Simple, mean-only M-PMC AIS version of MPOPI
"""
mutable struct μAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    λ_ais::Float64
    rng::R
    logger::MPPI_Logger
end

"""
μAISMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function μAISMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return μAISMPPI_Policy(params, env, U₀, Σ, opt_its, λ_ais, rng, mppi_logger)
end

"""
    calculate_trajectory_costs(policy::μAISMPPI_Policy, env::AbstractEnv)
Simple AIS strategy in which only the new mean is adapted at each
iteration. New samples are then taken from the new distribution.
"""
function calculate_trajectory_costs(pol::μAISMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    weight_method = Information_Theoretic(pol.λ_ais)

    U_orig = pol.U
    P = Distributions.MvNormal(pol.Σ)
    Σ_inv = Distributions.invcov(P)

    trajectory_cost = Vector{Float64}(undef, K) 
    ws = Vector{Float64}(undef, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        E = rand(pol.rng, P, K)
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            ws = compute_weights(weight_method, trajectory_cost)
            pw = StatsBase.ProbabilityWeights(ws)
            (μ′, Σ′) = StatsBase.mean_and_cov(E, pw, 2)
            pol.U = pol.U + vec(μ′)
        else
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E, ws
end

"""
μΣAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Mean and Covariance M-PMC AIS with one distribution version of MPOPI
"""
mutable struct μΣAISMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    λ_ais::Float64
    rng::R
    logger::MPPI_Logger
end

"""
μΣAISMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
    kwargs passed to MPPI_Policy_Params
"""
function μΣAISMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return μΣAISMPPI_Policy(params, env, U₀, Σ, opt_its, λ_ais, rng, mppi_logger)
end

"""
    calculate_trajectory_costs(policy::μΣAISMPPI_Policy, env::AbstractEnv)
Simple AIS strategy in which the new mean and covariance are adapted at each
iteration. New samples are then taken from the new distribution.
"""
function calculate_trajectory_costs(pol::μΣAISMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    weight_method = Information_Theoretic(pol.λ_ais)

    # Initial covariance of distribution
    U_orig = pol.U
    Σ′ = pol.Σ

    trajectory_cost = Vector{Float64}(undef, K) 
    ws = Vector{Float64}(undef, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            ws = compute_weights(weight_method, trajectory_cost)
            pw = StatsBase.ProbabilityWeights(ws)
            (μ′, Σ′) = StatsBase.mean_and_cov(E, pw, 2)
            Σ′ = Σ′ + + 10e-9*I
            pol.U = pol.U + vec(μ′)
        else
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E, ws
end

"""
PMCMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    PMC with multinomial resampling AIS with one distribution version of MPOPI
"""
mutable struct PMCMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    params::MPPI_Policy_Params
    env::AbstractEnv
    U::Vector{Float64}
    Σ::Matrix{Float64}
    opt_its::Int
    λ_ais::Float64
    rng::R
    logger::MPPI_Logger
end

"""
PMCMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function PMCMPPI_Policy(env::AbstractEnv; 
    opt_its::Int = 10,
    λ_ais::Float64 = 20.0,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    return PMCMPPI_Policy(params, env, U₀, Σ, opt_its, λ_ais, rng, mppi_logger)
end

"""
    calculate_trajectory_costs(policy::PMCMPPI_Policy, env::AbstractEnv)
Generic PMC strategy. 
O Cappé, A Guillin, J. M Marin & C. P Robert (2004) 
Population Monte Carlo, Journal of Computational and Graphical Statistics, 
13:4, 907-929, DOI: 10.1198/106186004X12803
"""
function calculate_trajectory_costs(pol::PMCMPPI_Policy, env::AbstractEnv)
    K = pol.params.num_samples
    N = pol.opt_its
    weight_method = Information_Theoretic(pol.λ_ais)

    # Initial covariance of distribution
    U_orig = pol.U
    Σ′ = pol.Σ

    trajectory_cost = Vector{Float64}(undef, K) 
    ws = Vector{Float64}(undef, K)
    # Optimize sample distribution and get trajectory costs
    for n ∈ 1:N
        # Get samples for which our trajectories will be defined
        P = Distributions.MvNormal(Σ′)
        E = rand(pol.rng, P, K)
        Σ_inv = Distributions.invcov(P)

        # Use the samples to simulate our model to get the costs
        trajectory_cost = simulate_model(pol, env, E, Σ_inv, U_orig)
        if n < N
            ws = compute_weights(weight_method, trajectory_cost)
            resample_cat_dist = Categorical(ws)
            resample_idxs = rand(pol.rng, resample_cat_dist, K)
            E′ = E[:, resample_idxs]
            (μ′, Σ′) = StatsBase.mean_and_cov(E′, 2)
            Σ′ = Σ′ + + 10e-9*I
            pol.U = pol.U + vec(μ′)
        else
            ws = compute_weights(pol.params.weight_method, trajectory_cost)
        end
    end
    E = E .+ (pol.U - U_orig)
    pol.U = U_orig
    return trajectory_cost, E, ws
end

"""
NESMPPI_Policy{R<:AbstractRNG} <: AbstractGMPPI_Policy
    Natual evolution strategy version of MPOPI
"""
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

"""
NESMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    step_factor::Float64 = 0.01,
    kwargs...
kwargs passed to MPPI_Policy_Params
"""
function NESMPPI_Policy(env::AbstractEnv;
    opt_its::Int = 10,
    step_factor::Float64 = 0.01,
    kwargs...
)
    params, U₀, Σ, rng, mppi_logger = MPPI_Policy_Params(env, :gmppi; kwargs...)
    A = sqrt(Σ)
    pol = NESMPPI_Policy(params, env, U₀, Σ, A, opt_its, step_factor, rng, mppi_logger)
    return pol
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
    weights = compute_weights(pol.params.weight_method, trajectory_cost)
    return trajectory_cost, E, weights
end





