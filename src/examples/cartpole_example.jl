
# Modifications to the RLBase functions to work with different GMPPI algorithms
function (env::CartPoleEnv)(a)
    length(a) == 1 || error("Only implented for 1 step")
    RLBase.act!(env, a[1])
end

""" 
simulate_cartpole(; kwargs...)
    Simulate a continuous cartpole envrionment
kwargs:
 - num_trials = 1,                                   # Number of trials
 - num_steps = 200,                                  # Max number of steps per scenario
 - policy_type = :cemppi,                            # Type of policy (see `get_policy` for options)
 - num_samples = 20,                                 # Number of initial samples for the policy
 - horizon = 15,                                     # Time step horizon
 - λ = 0.1,                                          # Inverve temperature settings for IT weighting
 - α = 1.0,                                          # Control cost parameter
 - U₀ = [0.0],                                       # Default initial contorl
 - cov_mat = [1.5]                                   # Control covariance matrix
 - ais_its = 5,                                      # Number of AIS algo iterations
 - λ_ais = 0.1,                                      # Inverse temperature for AIS algo (if applicable)
 - ce_elite_threshold = 0.8,                         # CE elite threshold (if applicable)
 - ce_Σ_est = :mle,                                  # CE Σ estimation methods (if applicable)
 - cma_σ = 0.75,                                     # CMA step factor (if applicable)
 - cma_elite_threshold = 0.8,                        # CMA elite threshold (if applicable)
 - seed = Int(rand(1:10e10)),                        # Seed algorithm and envrionment (incremtented by trial number)
 - log_runs = true,                                  # Output results at each of each trial
 - plot_steps = false,                               # Plot each step (NOT RECOMMENDED FOR LARGE RUNS)
 - pol_log = false,                                  # Turn on policy logger (slows down the process)
 - save_gif = false,                                 # Save gif
"""
function simulate_cartpole(;
    num_trials = 1,
    num_steps = 200,
    policy_type = :cemppi,
    num_samples = 20, 
    horizon = 15,
    λ = 0.1,
    α = 1.0,
    U₀ = [0.0],
    cov_mat = [1.5],
    ais_its = 5,
    λ_ais = 0.1,
    ce_elite_threshold = 0.8,
    ce_Σ_est = :mle,
    cma_σ = 0.75,
    cma_elite_threshold = 0.8,
    seed = Int(rand(1:10e10)),
    log_runs = true,
    plot_steps = false,
    pol_log = false,
    save_gif = false,
)

    sim_type = "CartPole"

    @printf("\n")
    @printf("%-30s%s\n", "Sim Type:", sim_type)
    @printf("%-30s%d\n", "Num Trails:", num_trials)
    @printf("%-30s%d\n", "Num Steps:", num_steps)
    @printf("%-30s%s\n","Policy Type:", policy_type)
    @printf("%-30s%d\n", "Num samples", num_samples)
    @printf("%-30s%d\n", "Horizon", horizon)
    @printf("%-30s%.2f\n", "λ (inverse temp):", λ)
    @printf("%-30s%.2f\n", "α (control cost param):", α)
    if policy_type != :mppi && policy_type != :gmppi
        @printf("%-30s%d\n", "# AIS Iterations:", ais_its)
        if policy_type ∈ [:μΣaismppi, :μaismppi, :pmcmppi]
            @printf("%-30s%.2f\n", "λ_ais (ais inverse temp):", λ_ais)
        elseif policy_type == :cemppi
            @printf("%-30s%.2f\n", "CE Elite Threshold:", ce_elite_threshold)
            @printf("%-30s%s\n", "CE Σ Est Method:", ce_Σ_est)
        elseif policy_type == :cmamppi
            @printf("%-30s%.2f\n", "CMA Step Factor (σ):", cma_σ)
            @printf("%-30s%.2f\n", "CMA Elite Perc Thres:", cma_elite_threshold)
        end
    end
    @printf("%-30s[%.4f, ..., %.4f]\n", "U₀", U₀[1], U₀[end])
    @printf("%-30s[%.4f]\n", "Σ", cov_mat[1,1])
    @printf("%-30s%d\n", "Seed:", seed)
    @printf("\n")
    
    gif_name = "$sim_type-$policy_type-$num_samples-$horizon-$λ-$α-"
    if policy_type != :mppi && policy_type != :gmppi
        gif_name = gif_name * "$ais_its-"
    end
    if policy_type == :cemppi
        gif_name = gif_name * "$ce_elite_threshold-"
        gif_name = gif_name * "$ce_Σ_est-"
    elseif policy_type ∈ [:μΣaismppi, :μaismppi, :pmcmppi]
        gif_name = gif_name * "$λ_ais-"
    elseif policy_type == :cmamppi
        gif_name = gif_name * "$cma_σ-"
        gif_name = gif_name * "$cma_elite_threshold-"
    end
    gif_name = gif_name * "$num_trials.gif"
    anim = Animation()

    rews = zeros(Float64, num_trials)
    steps = zeros(Float64, num_trials)
    rews_per_step = zeros(Float64, num_trials)
    exec_times = zeros(Float64, num_trials)  

    @printf("Trial    #: %12s : %7s: %12s", "Reward", "Steps", "Reward/Step")
    @printf(" : %7s", "Ex Time")
    @printf("\n")

    for k ∈ 1:num_trials
        env = CartPoleEnv(continuous=true, rng=MersenneTwister())
        pol = get_policy(
            policy_type,
            env,num_samples, horizon, λ, α, U₀, cov_mat, pol_log, 
            ais_its, 
            λ_ais, 
            ce_elite_threshold, ce_Σ_est,
            cma_σ, cma_elite_threshold,  
        )

        seed!(env, seed + k)
        seed!(pol, seed + k)

        # Start timer
        time_start = Dates.now()
        
        rew, cnt = 0, 0
        # Main simulation loop
        while !env.done && cnt <= num_steps
            # Get action from policy
            act = pol(env)
            # Apply action to envrionment
            env(act)
            cnt += 1
            # Ger reward at the step
            step_rew = reward(env)
            rew += step_rew

            # Plot or collect the plot for the animation
            if plot_steps || save_gif
                p = plot(env)
                if save_gif frame(anim) end
                if plot_steps display(p) end
            end
        end
        
        # Stop timer
        time_end = Dates.now()
        seconds_ran = Dates.value(time_end - time_start) / 1000

        rews[k] = rew
        steps[k] = cnt-1
        rews_per_step[k] = rews[k]/steps[k]
        exec_times[k] = seconds_ran 

        if log_runs
            @printf("Trial %4d: %12.2f : %7d: %12.2f", k, rew, cnt-1, rew/(cnt-1))
            @printf(" : %7.2f", seconds_ran)
            @printf("\n")
        end
    end

    # Output summary results
    @printf("-----------------------------------\n")
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "AVE", mean(rews), mean(steps), mean(rews_per_step))
    @printf(" : %7.2f\n", mean(exec_times))    
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "STD", std(rews), std(steps), std(rews_per_step))
    @printf(" : %7.2f\n", std(exec_times))
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MED", 
        quantile_ci(rews)[2], quantile_ci(steps)[2], quantile_ci(rews_per_step)[2])
    @printf(" : %7.2f\n", quantile_ci(exec_times)[2])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "L95", 
        quantile_ci(rews)[1], quantile_ci(steps)[1], quantile_ci(rews_per_step)[1])
    @printf(" : %7.2f\n", quantile_ci(exec_times)[1])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "U95", 
        quantile_ci(rews)[3], quantile_ci(steps)[3], quantile_ci(rews_per_step)[3])
    @printf(" : %7.2f\n", quantile_ci(exec_times)[3])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MIN", 
        minimum(rews), minimum(steps), minimum(rews_per_step))
    @printf(" : %7.2f\n", minimum(exec_times))
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MAX", maximum(rews), maximum(steps), maximum(rews_per_step))
    @printf(" : %7.2f\n", maximum(exec_times))

    if save_gif
        println("Saving gif...$gif_name")
        gif(anim, gif_name, fps=10)
    end
end
