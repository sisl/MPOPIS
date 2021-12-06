
"""
Example simulating the car racing environment.

Dependencies:
 - MPOPIS
 - Printf
 - Random
 - Plots
 - ProgressMeter
 - Dates
 - LinearAlgebra
 - Distributions
 - example_utils.jl
"""

""" 
simulate_car_racing(; kwargs...)
    Simulate a car racing scenarion with 1 or multiple cars
kwargs:
 - num_trials = 1,                                   # Number of trials
 - num_steps = 200,                                  # Max number of steps per scenario
 - num_cars = 1,                                     # Number of cars
 - laps = 2,                                         # Max number of laps (if using curve.csv track)
 - policy_type = :cemppi,                            # Type of policy (see `get_policy` for options)
 - num_samples = 150,                                # Number of initial samples for the policy
 - horizon = 50,                                     # Time step horizon
 - λ = 10.0,                                         # Inverve temperature settings for IT weighting
 - α = 1.0,                                          # Control cost parameter
 - U₀ = zeros(Float64, num_cars*2),                  # Default initial contorl
 - cov_mat = block_diagm([0.0625, 0.1], num_cars),   # Control covariance matrix
 - ais_its = 10,                                     # Number of AIS algo iterations
 - λ_ais = 20.0,                                     # Inverse temperature for AIS algo (if applicable)
 - ce_elite_threshold = 0.8,                         # CE elite threshold (if applicable)
 - ce_Σ_est = :ss,                                   # CE Σ estimation methods (if applicable)
 - cma_σ = 0.75,                                     # CMA step factor (if applicable)
 - cma_elite_threshold = 0.8,                        # CMA elite threshold (if applicable)
 - state_x_sigma = 0.0,                              # Add normal noise std dev to x position at each time step (only for single car)
 - state_y_sigma = 0.0,                              # Add normal noise std dev to y position at each time step (only for single car)
 - state_ψ_sigma = 0.0,                              # Add normal noise std dev to heading at each time step (only for single car)
 - seed = Int(rand(1:10e10)),                        # Seed algorithm and envrionment (incremtented by trial number)
 - log_runs = true,                                  # Output results at each of each trial
 - plot_steps = false,                               # Plot each step (NOT RECOMMENDED FOR LARGE RUNS)
 - pol_log = false,                                  # Turn on policy logger (slows down the process)
 - plot_traj = false,                                # Plot the samples trajectories
 - plot_traj_perc = 1.0,                             # Percentage of samples trajectories to plot (if applicable)
 - text_with_plot = true,                            # Plot car 1 state info on plot
 - text_on_plot_xy = (80.0, -60.0)                   # XY position of output text (if applicable)
 - save_gif = false,                                 # Save gif
"""
function simulate_car_racing(;
    num_trials = 1,
    num_steps = 200,
    num_cars = 1,
    policy_type = :cemppi,
    laps = 2,
    num_samples = 150, 
    horizon = 50,
    λ = 10.0,
    α = 1.0,
    U₀ = zeros(Float64, num_cars*2),
    cov_mat = block_diagm([0.0625, 0.1], num_cars),
    ais_its = 10,
    λ_ais = 20.0,
    ce_elite_threshold = 0.8,
    ce_Σ_est = :ss,
    cma_σ = 0.75,
    cma_elite_threshold = 0.8,
    state_x_sigma = 0.0,
    state_y_sigma = 0.0,
    state_ψ_sigma = 0.0,
    seed = Int(rand(1:10e10)),
    log_runs = true,
    plot_steps = false,
    pol_log = false,
    plot_traj = false,
    plot_traj_perc = 1.0,
    text_with_plot = true,
    text_on_plot_xy = (80.0, -60.0),
    save_gif = false,
)

    if num_cars > 1
        sim_type = :mcr
    else
        sim_type = :cr
    end

    @printf("\n")
    @printf("%-30s%s\n", "Sim Type:", sim_type)
    @printf("%-30s%d\n", "Num Cars:", num_cars)
    @printf("%-30s%d\n", "Num Trails:", num_trials)
    @printf("%-30s%d\n", "Num Steps:", num_steps)
    @printf("%-30s%d\n", "Max Num Laps:", laps)
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
    @printf("%-30s%s([%.4f %.4f; %.4f %.4f], %d)\n", "Σ", "block_diagm",
        cov_mat[1,1], cov_mat[1,2], cov_mat[2,1], cov_mat[2,2], num_cars)
    if num_cars == 1
        @printf("%-30s%.4f\n", "Noise, State X σ:", state_x_sigma)
        @printf("%-30s%.4f\n", "Noise, State Y σ:", state_y_sigma)
        @printf("%-30s%.4f\n", "Noise, Heading σ:", state_ψ_sigma)
    end
    @printf("%-30s%d\n", "Seed:", seed)
    @printf("\n")
    
    # Must of policy log on if pltting trajectories
    if plot_traj
        pol_log = true
    end
    
    gif_name = "$sim_type-$num_cars-$policy_type-$num_samples-$horizon-$λ-$α-"
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
    gif_name = gif_name * "$num_trials-$laps.gif"
    anim = Animation()

    rews = zeros(Float64, num_trials)
    steps = zeros(Float64, num_trials)
    rews_per_step = zeros(Float64, num_trials)
    lap_ts = [zeros(Float64, num_trials) for _ in 1:laps]
    mean_vs = zeros(Float64, num_trials)
    max_vs = zeros(Float64, num_trials)
    mean_βs = zeros(Float64, num_trials)
    max_βs = zeros(Float64, num_trials)
    β_viols = zeros(Float64, num_trials)
    T_viols = zeros(Float64, num_trials)
    C_viols = zeros(Float64, num_trials)
    exec_times = zeros(Float64, num_trials)  

    @printf("Trial    #: %12s : %7s: %12s", "Reward", "Steps", "Reward/Step")
    for ii ∈ 1:laps
        @printf(" : %6s%d", "lap ", ii)
    end
    @printf(" : %7s : %7s", "Mean V", "Max V")
    @printf(" : %7s : %7s", "Mean β", "Max β")
    @printf(" : %7s : %7s", "β Viol", "T Viol")
    if sim_type == :mcr
        @printf(" : %7s", "C Viol")
    end
    @printf(" : %7s", "Ex Time")
    @printf("\n")

    for k ∈ 1:num_trials
        
        if sim_type == :cr
            env = CarRacingEnv(rng=MersenneTwister())
        elseif sim_type == :mcr
            env = MultiCarRacingEnv(num_cars, rng=MersenneTwister())
        end

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

        pm = Progress(num_steps, 1, "Trial $k ....", 50)
        # Start timer
        time_start = Dates.now()
        
        lap_time = zeros(Int, laps)
        v_mean_log = Vector{Float64}()
        v_max_log = Vector{Float64}()
        β_mean_log = Vector{Float64}()
        β_max_log = Vector{Float64}()
        rew, cnt, lap, prev_y = 0, 0, 0, 0
        trk_viol, β_viol, crash_viol = 0, 0, 0

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
                if plot_traj
                    p = plot(env, pol, plot_traj_perc, text_output=text_with_plot, text_xy=text_on_plot_xy)
                else 
                    p = plot(env, text_output=text_with_plot, text_xy=text_on_plot_xy)
                end
                if save_gif frame(anim) end
                if plot_steps display(p) end
            end

            if sim_type == :cr
                env.state[1] += state_x_sigma * randn(env.rng)
                env.state[2] += state_y_sigma * randn(env.rng)

                δψ = state_ψ_sigma * randn(env.rng)
                env.state[3] += δψ
                
                # Passive rotation matrix
                rot_mat = [ cos(δψ) sin(δψ) ;
                           -sin(δψ) cos(δψ) ]
                V′ = rot_mat*[env.state[4]; env.state[5]]
                env.state[4:5] = V′
            end

            next!(pm)

            # Get logging information
            curr_y = env.state[2]
            if sim_type == :mcr
                curr_y = minimum([en.state[2] for en ∈ env.envs])    
                vs = [norm(en.state[4:5]) for en ∈ env.envs]
                βs = [abs(calculate_β(en)) for en ∈ env.envs]
            else
                vs = norm(env.state[4:5])
                βs = abs(calculate_β(env))
            end
            push!(v_mean_log, mean(vs))
            push!(v_max_log, maximum(vs))
            push!(β_mean_log, mean(βs))
            push!(β_max_log, maximum(βs))
            
            # Determine if violations occurred
            if step_rew < -4000
                ex_β = exceed_β(env)
                within_t = sim_type == :cr ? within_track(env).within : within_track(env)
                if ex_β β_viol += 1 end
                if !within_t trk_viol += 1 end
                temp_rew = step_rew + ex_β*5000 + !within_t*1000000
                if temp_rew < -10500 crash_viol += 1 end
            end

            if sim_type == :mcr
                # Not exact, but works
                d = minimum([norm(en.state[1:2]) for en ∈ env.envs])
            else
                d = norm(env.state[1:2])
            end

            # Estimate to increment lap count on curve.csv
            if prev_y < 0.0 && curr_y >= 0.0 && d <= 15.0
                lap += 1
                lap_time[lap] = cnt
            end
            if lap >= laps || trk_viol > 10 || β_viol > 50
                env.done = true
            end
            prev_y = curr_y
        end
        
        # Stop timer
        time_end = Dates.now()
        seconds_ran = Dates.value(time_end - time_start) / 1000

        rews[k] = rew
        steps[k] = cnt-1
        rews_per_step[k] = rews[k]/steps[k]
        exec_times[k] = seconds_ran 
        if sim_type ∈ (:cr, :mcr)
            for ii ∈ 1:laps
                lap_ts[ii][k] = lap_time[ii]
            end
            mean_vs[k] = mean(v_mean_log)
            max_vs[k] = maximum(v_max_log)
            mean_βs[k] = mean(β_mean_log)
            max_βs[k] = maximum(β_max_log)
            β_viols[k] = β_viol
            T_viols[k] = trk_viol
            C_viols[k] = crash_viol
        end 

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
            for ii ∈ 1:laps
                @printf(" : %7d", lap_time[ii])
            end
            @printf(" : %7.2f : %7.2f", mean(v_mean_log), maximum(v_max_log))
            @printf(" : %7.2f : %7.2f",  mean(β_mean_log), maximum(β_max_log))
            @printf(" : %7d : %7d", β_viol, trk_viol)
            if sim_type == :mcr
                @printf(" : %7d", crash_viol)
            end
            @printf(" : %7.2f", seconds_ran)
            @printf("\n")
        end
    end

    # Output summary results
    @printf("-----------------------------------\n")
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "AVE", mean(rews), mean(steps), mean(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", mean(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", mean(mean_vs), mean(max_vs))
    @printf(" : %7.2f : %7.2f",  mean(mean_βs), mean(max_βs))
    @printf(" : %7.2f : %7.2f", mean(β_viols), mean(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", mean(C_viols))
    end
    @printf(" : %7.2f\n", mean(exec_times))    
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "STD", std(rews), std(steps), std(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", std(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", std(mean_vs), std(max_vs))
    @printf(" : %7.2f : %7.2f",  std(mean_βs), std(max_βs))
    @printf(" : %7.2f : %7.2f", std(β_viols), std(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", std(C_viols))
    end
    @printf(" : %7.2f\n", std(exec_times))
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MED", 
        quantile_ci(rews)[2], quantile_ci(steps)[2], quantile_ci(rews_per_step)[2])
    for ii ∈ 1:laps
        @printf(" : %7.2f", quantile_ci(lap_ts[ii])[2])
    end
    @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[2], quantile_ci(max_vs)[2])
    @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[2], quantile_ci(max_βs)[2])
    @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[2], quantile_ci(T_viols)[2])
    if sim_type == :mcr
        @printf(" : %7.2f", quantile_ci(C_viols)[2])
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[2])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "L95", 
        quantile_ci(rews)[1], quantile_ci(steps)[1], quantile_ci(rews_per_step)[1])
    for ii ∈ 1:laps
        @printf(" : %7.2f", quantile_ci(lap_ts[ii])[1])
    end
    @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[1], quantile_ci(max_vs)[1])
    @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[1], quantile_ci(max_βs)[1])
    @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[1], quantile_ci(T_viols)[1])
    if sim_type == :mcr
        @printf(" : %7.2f", quantile_ci(C_viols)[1])
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[1])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "U95", 
        quantile_ci(rews)[3], quantile_ci(steps)[3], quantile_ci(rews_per_step)[3])
    for ii ∈ 1:laps
        @printf(" : %7.2f", quantile_ci(lap_ts[ii])[3])
    end
    @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[3], quantile_ci(max_vs)[3])
    @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[3], quantile_ci(max_βs)[3])
    @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[3], quantile_ci(T_viols)[3])
    if sim_type == :mcr
        @printf(" : %7.2f", quantile_ci(C_viols)[3])
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[3])
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MIN", 
        minimum(rews), minimum(steps), minimum(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", minimum(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", minimum(mean_vs), minimum(max_vs))
    @printf(" : %7.2f : %7.2f",  minimum(mean_βs), minimum(max_βs))
    @printf(" : %7.2f : %7.2f", minimum(β_viols), minimum(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", minimum(C_viols))
    end
    @printf(" : %7.2f\n", minimum(exec_times))
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MAX", maximum(rews), maximum(steps), maximum(rews_per_step))
    for ii ∈ 1:laps
        @printf(" : %7.2f", maximum(lap_ts[ii]))
    end
    @printf(" : %7.2f : %7.2f", maximum(mean_vs), maximum(max_vs))
    @printf(" : %7.2f : %7.2f",  maximum(mean_βs), maximum(max_βs))
    @printf(" : %7.2f : %7.2f", maximum(β_viols), maximum(T_viols))
    if sim_type == :mcr
        @printf(" : %7.2f", maximum(C_viols))
    end
    @printf(" : %7.2f\n", maximum(exec_times))

    if save_gif
        println("Saving gif...$gif_name")
        gif(anim, gif_name, fps=10)
    end
end

