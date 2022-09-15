using Revise
using MPOPIS
using ProgressMeter
using Random

function get_policy(
    policy_type,
    env, num_samples, horizon, λ, α, U₀, cov_mat, pol_log,
    ais_its,
    λ_ais,
    ce_elite_threshold, ce_Σ_est,
    cma_σ, cma_elite_threshold,
)
    if  policy_type == :mppi
        pol = MPPI_Policy(
            env,
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            log=pol_log,
            rng=MersenneTwister(),
            )
    elseif policy_type == :gmppi
        pol = GMPPI_Policy(
            env,
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :cemppi
        pol = CEMPPI_Policy(
            env,
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=ais_its,
            ce_elite_threshold=ce_elite_threshold,
            Σ_est = ce_Σ_est,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :cmamppi
        pol = CMAMPPI_Policy(
            env,
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=ais_its,
            σ=cma_σ,
            elite_perc_threshold=cma_elite_threshold,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :μΣaismppi
        pol = μΣAISMPPI_Policy(env,
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=ais_its,
            λ_ais=λ_ais,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :μaismppi
        pol = μAISMPPI_Policy(env,
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=ais_its,
            λ_ais=λ_ais,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :pmcmppi
        pol = PMCMPPI_Policy(env,
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=ais_its,
            λ_ais=λ_ais,
            log=pol_log,
            rng=MersenneTwister(),
        )
    else
        error("No policy_type of $policy_type")
    end
    return pol
end

num_cars = 1

# for nn = 1:60

    num_steps = 30
    laps = 2

    policy_type = :cemppi
    num_samples = 150
    ais_its = 10

    horizon = 50
    λ = 10.0
    α = 1.0

    U₀ = zeros(Float64, num_cars*2)
    cov_mat = block_diagm([0.0625, 0.1], num_cars)

    λ_ais = 70.0

    ce_elite_threshold = 0.8
    ce_Σ_est = :ss

    cma_σ = 0.75
    cma_elite_threshold = 0.8
    seed = 1

    log_runs = true
    plot_traj_perc = 1.0
    text_with_plot = false
    pol_log = true

    if num_cars > 1
        sim_type = :mcr
    else
        sim_type = :cr
    end


    if sim_type == :cr
        env = CarRacingEnv(rng=MersenneTwister())
    elseif sim_type == :mcr
        env = MultiCarRacingEnv(num_cars, rng=MersenneTwister())
    end

    pols = Vector(undef, ais_its)
    for p_i = 1:ais_its
        pols[p_i] = get_policy(
            policy_type,
            env,num_samples, horizon, λ, α, U₀, cov_mat, pol_log,
            p_i,
            λ_ais,
            ce_elite_threshold, ce_Σ_est,
            cma_σ, cma_elite_threshold,
        )
    end

    seed!(env, seed)
    for p_i = 1:ais_its
        seed!(pols[p_i], seed)
    end

    pm = Progress(num_steps)

    # Main simulation loop
    for ii ∈ 1:num_steps
        # Get action from policy
        act = pols[end](env)
        # Apply action to envrionment
        env(act)

        for p_i = 1:(ais_its-1)
            pols[p_i].U = copy(pols[end].U)
            pols[p_i].env = copy(pols[end].env)
        end
        next!(pm)
    end


    ps = []
    for p_i = 1:ais_its
        env_c = copy(env)
        act = pols[p_i](env_c)
        env_c(act)
        p = plot(env_c, pols[p_i], plot_traj_perc, text_output=text_with_plot)
        p = plot(p, xlim=(-15, 82), ylim=(35, 156), grid=false, foreground_color_axis=:white)
        push!(ps, p)
    end



    # for p ∈ ps
    #     display(p)
    # end
# end
