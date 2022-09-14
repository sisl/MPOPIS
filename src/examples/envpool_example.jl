function simulate_envpool_env(
    env_name;
    frame_skip = 10,

    num_trials = 1,
    num_steps = 200,

    policy_type = :cemppi,
    num_samples = 150,
    horizon = 50,
    λ = 1.0,
    α = 1.0,

    U₀ = [],
    cov_mat = [],

    ais_its = 10,
    λ_ais = 20.0,
    ce_elite_threshold = 0.8,
    ce_Σ_est = :ss,
    cma_σ = 0.75,
    cma_elite_threshold = 0.8,

    seed = Int(rand(1:10e10)),
    plot_steps = false,
    log_runs = true,
    pol_log = false,
    save_gif = false,
)

    env = EnvpoolEnv(env_name; num_envs=1)
    as = length(action_space(env).left)
    if isempty(U₀)
        U₀ = zeros(as)
    end
    if isempty(cov_mat)
        cov_mat = Matrix(I(as)*0.5)
    end

    @printf("\n")
    @printf("%-30s%s\n", "Env Name:", env_name)
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
    @printf("%-30s", "Σ")
    cov_mat_diag = diag(cov_mat)
    for cov_mat_dᵢ ∈ cov_mat_diag
        @printf("%.4f ", cov_mat_dᵢ)
    end
    @printf("\n")
    @printf("%-30s%d\n", "Seed:", seed)
    @printf("\n")

    gif_name = "$env_name-$policy_type-$num_samples-$horizon-$λ-$α-"
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

        env = EnvpoolEnv(
            env_name;
            frame_skip=frame_skip,
            num_envs=num_samples,
            rng=MersenneTwister()
        )

        pol = get_policy(
            policy_type,
            env,
            num_samples,
            horizon,
            λ,
            α,
            U₀,
            cov_mat,
            pol_log,
            ais_its,
            λ_ais,
            ce_elite_threshold,
            ce_Σ_est,
            cma_σ,
            cma_elite_threshold,
        )

        seed!(env, seed + k)
        seed!(pol, seed + k)

        pm = Progress(num_steps, 1, "Trial $k ....", 50)
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
            # Get reward at the step
            step_rew = reward(env)[1]
            rew += step_rew

            # # Plot or collect the plot for the animation
            # if plot_steps || save_gif

            #     p = plot(env, text_output=text_with_plot, text_xy=text_on_plot_xy)

            #     if save_gif frame(anim) end
            #     if plot_steps display(p) end
            # end

            next!(pm)
        end

        # Stop timer
        time_end = Dates.now()
        seconds_ran = Dates.value(time_end - time_start) / 1000

        rews[k] = rew
        steps[k] = cnt-1
        rews_per_step[k] = rews[k]/steps[k]
        exec_times[k] = seconds_ran

        # For clearing the progress bar
        if cnt > num_steps
            print("\u1b[1F") # Moves cursor to beginning of the line n lines up
            print("\u1b[0K") # Clears  part of the line. n=0: clear from cursor to end
        else
            print("\e[2K") # clear whole line
            print("\e[1G") # move cursor to column 1
        end

        if log_runs
            @printf("Trial %4d: %12.2f : %7d: %12.2f", k, rew, cnt-1, rew/(cnt-1))
            @printf(" : %7.2f", seconds_ran)
            @printf("\n")
        end
        # env.py_env.close()
        env = nothing
        pol = nothing
        GC.gc()
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

    # if save_gif
    #     println("Saving gif...$gif_name")
    #     gif(anim, gif_name, fps=10)
    # end
end
