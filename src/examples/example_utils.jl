
function quantile_ci(x, p=0.05, q=0.5)
    n = length(x)
    zm = quantile(Normal(0.0, 1.0), p/2)
    zp = quantile(Normal(0.0, 1.0), 1-p/2)
    j = max(Int(ceil(n*q + zm*sqrt(n*q*(1-q)))), 1)
    k = min(Int(ceil(n*q + zp*sqrt(n*q*(1-q)))), length(x))
    x_sorted = sort(x)
    return x_sorted[j], quantile(x, q), x_sorted[k]
end

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