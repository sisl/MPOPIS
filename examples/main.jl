
using Printf
using ProgressMeter
using Dates

using Random
using Plots

using MPOPIS


function simulate_environment(environment;
    num_trials = 1,
    num_steps = 200,
    policy_type = :gmppi,
    plot_steps = false,
    save_gif = false,
    plot_traj = false,
    plot_traj_perc = 1.0,
    laps = 2,
    num_cars = 2,
    continuous = true,
    num_samples = 10, 
    horizon = 15,
    λ = 1.0,
    α = 1.0,
    U₀ = [0.0],
    cov_mat = [1.5],
    opt_its = 10,
    λ_ais = 20.0,
    ce_elite_threshold = 0.8,
    min_max_sample_perc = 0.1,
    step_factor = 0.01,
    σ = 1.0,
    elite_perc_threshold = 0.8,
    pol_log = false,
    seeded=nothing,
    state_x_sigma=0.0,
    state_y_sigma=0.0,
    state_ψ_sigma=0.0,
    Σ_est=:mle,
)

    gif_name = "$environment-$num_cars-$policy_type-$num_samples-$horizon-$λ-$α-"
    cov_v = cov_mat[1,1]
    gif_name = gif_name * "$cov_v-"
    if policy_type == :cemppi
        gif_name = gif_name * "$opt_its-$ce_elite_threshold-"
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
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %6s%d", "lap ", ii)
        end
        @printf(" : %7s : %7s", "Mean V", "Max V")
        @printf(" : %7s : %7s", "Mean β", "Max β")
        @printf(" : %7s : %7s", "β Viol", "T Viol")
        if environment == :mcr
            @printf(" : %7s", "C Viol")
        end
    end
    @printf(" : %7s", "Ex Time")
    @printf("\n")
    for k ∈ 1:num_trials
        # Start timer
        time_start = Dates.now()

        env = get_environment(environment, 
            continuous=continuous, num_cars=num_cars);
        pol = get_policy(
                env, policy_type, num_samples, horizon,
                λ, α, U₀, cov_mat, opt_its, ce_elite_threshold, 
                step_factor, σ, elite_perc_threshold,  pol_log, 
                Σ_est, λ_ais,
        )
        if isnothing(seeded)
            seed!(env, k)
            seed!(pol, k)
        else
            seed!(env, seeded + k - 1)
            seed!(pol, seeded + k - 1)
        end
        
        pm = Progress(num_steps, 1, "Trial $k ....", 50)
        
        lap_time = zeros(Int, laps)
        v_mean_log = Vector{Float64}()
        v_max_log = Vector{Float64}()
        β_mean_log = Vector{Float64}()
        β_max_log = Vector{Float64}()
        rew, cnt, lap, prev_y = 0, 0, 0, 0
        trk_viol, β_viol, crash_viol = 0, 0, 0
        while !env.done && cnt <= num_steps
            act = pol(env)
            env(act)
            cnt += 1
            step_rew = reward(env)
            rew += step_rew
            if plot_steps || save_gif
                if plot_traj && pol_log
                    p = plot(env, pol, plot_traj_perc)
                else 
                    p = plot(env)
                end
                if save_gif frame(anim) end
                if plot_steps display(p) end
            end

            if environment == :cr
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

            if environment ∈ (:cr, :mcr)
                curr_y = env.state[2]
                if environment == :mcr
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
                
                if step_rew < -4000
                    ex_β = exceed_β(env)
                    within_t = environment == :cr ? within_track(env).within : within_track(env)
                    if ex_β β_viol += 1 end
                    if !within_t trk_viol += 1 end
                    temp_rew = step_rew + ex_β*5000 + !within_t*1000000
                    if temp_rew < -10500 crash_viol += 1 end
                end

                if environment == :mcr
                    # Not exact, but should work
                    d = minimum([norm(en.state[1:2]) for en ∈ env.envs])
                else
                    d = norm(env.state[1:2])
                end

                if prev_y < 0.0 && curr_y >= 0.0 && d <= 15.0
                    lap += 1
                    lap_time[lap] = cnt
                end
                if lap >= laps || trk_viol > 10 || β_viol > 50
                    env.done = true
                end
                prev_y = curr_y
            end
        end

        if cnt > num_steps
            print("\u1b[1F") # Moves cursor to beginning of the line n lines up 
            print("\u1b[0K") # Clears  part of the line. n=0: clear from cursor to end
        else
            print("\e[2K") # clear whole line
            print("\e[1G") # move cursor to column 1
        end

        @printf("Trial %4d: %12.2f : %7d: %12.2f", k, rew, cnt-1, rew/(cnt-1))
        if environment ∈ (:cr, :mcr)
            for ii ∈ 1:laps
                @printf(" : %7d", lap_time[ii])
            end
            @printf(" : %7.2f : %7.2f", mean(v_mean_log), maximum(v_max_log))
            @printf(" : %7.2f : %7.2f",  mean(β_mean_log), maximum(β_max_log))
            @printf(" : %7d : %7d", β_viol, trk_viol)
            if environment == :mcr
                @printf(" : %7d", crash_viol)
            end
        end

        time_end = Dates.now()
        seconds_ran = Dates.value(time_end - time_start) / 1000
        @printf(" : %7.2f", seconds_ran)
        @printf("\n")

        rews[k] = rew
        steps[k] = cnt-1
        rews_per_step[k] = rews[k]/steps[k]
        exec_times[k] = seconds_ran 
        if environment ∈ (:cr, :mcr)
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

    end

    @printf("-----------------------------------\n")

    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "AVE", mean(rews), mean(steps), mean(rews_per_step))
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %7.2f", mean(lap_ts[ii]))
        end
        @printf(" : %7.2f : %7.2f", mean(mean_vs), mean(max_vs))
        @printf(" : %7.2f : %7.2f",  mean(mean_βs), mean(max_βs))
        @printf(" : %7.2f : %7.2f", mean(β_viols), mean(T_viols))
        if environment == :mcr
            @printf(" : %7.2f", mean(C_viols))
        end
    end
    @printf(" : %7.2f\n", mean(exec_times))
    
    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "STD", std(rews), std(steps), std(rews_per_step))
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %7.2f", std(lap_ts[ii]))
        end
        @printf(" : %7.2f : %7.2f", std(mean_vs), std(max_vs))
        @printf(" : %7.2f : %7.2f",  std(mean_βs), std(max_βs))
        @printf(" : %7.2f : %7.2f", std(β_viols), std(T_viols))
        if environment == :mcr
            @printf(" : %7.2f", std(C_viols))
        end
    end
    @printf(" : %7.2f\n", std(exec_times))

    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MED", 
        quantile_ci(rews)[2], quantile_ci(steps)[2], quantile_ci(rews_per_step)[2])
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %7.2f", quantile_ci(lap_ts[ii])[2])
        end
        @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[2], quantile_ci(max_vs)[2])
        @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[2], quantile_ci(max_βs)[2])
        @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[2], quantile_ci(T_viols)[2])
        if environment == :mcr
            @printf(" : %7.2f", quantile_ci(C_viols)[2])
        end
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[2])

    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "L95", 
        quantile_ci(rews)[1], quantile_ci(steps)[1], quantile_ci(rews_per_step)[1])
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %7.2f", quantile_ci(lap_ts[ii])[1])
        end
        @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[1], quantile_ci(max_vs)[1])
        @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[1], quantile_ci(max_βs)[1])
        @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[1], quantile_ci(T_viols)[1])
        if environment == :mcr
            @printf(" : %7.2f", quantile_ci(C_viols)[1])
        end
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[1])

    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "U95", 
        quantile_ci(rews)[3], quantile_ci(steps)[3], quantile_ci(rews_per_step)[3])
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %7.2f", quantile_ci(lap_ts[ii])[3])
        end
        @printf(" : %7.2f : %7.2f", quantile_ci(mean_vs)[3], quantile_ci(max_vs)[3])
        @printf(" : %7.2f : %7.2f",  quantile_ci(mean_βs)[3], quantile_ci(max_βs)[3])
        @printf(" : %7.2f : %7.2f", quantile_ci(β_viols)[3], quantile_ci(T_viols)[3])
        if environment == :mcr
            @printf(" : %7.2f", quantile_ci(C_viols)[3])
        end
    end
    @printf(" : %7.2f\n", quantile_ci(exec_times)[3])

    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MIN", 
        minimum(rews), minimum(steps), minimum(rews_per_step))
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %7.2f", minimum(lap_ts[ii]))
        end
        @printf(" : %7.2f : %7.2f", minimum(mean_vs), minimum(max_vs))
        @printf(" : %7.2f : %7.2f",  minimum(mean_βs), minimum(max_βs))
        @printf(" : %7.2f : %7.2f", minimum(β_viols), minimum(T_viols))
        if environment == :mcr
            @printf(" : %7.2f", minimum(C_viols))
        end
    end
    @printf(" : %7.2f\n", minimum(exec_times))

    @printf("Trials %3s: %12.2f : %7.2f: %12.2f", "MAX", maximum(rews), maximum(steps), maximum(rews_per_step))
    if environment ∈ (:cr, :mcr)
        for ii ∈ 1:laps
            @printf(" : %7.2f", maximum(lap_ts[ii]))
        end
        @printf(" : %7.2f : %7.2f", maximum(mean_vs), maximum(max_vs))
        @printf(" : %7.2f : %7.2f",  maximum(mean_βs), maximum(max_βs))
        @printf(" : %7.2f : %7.2f", maximum(β_viols), maximum(T_viols))
        if environment == :mcr
            @printf(" : %7.2f", maximum(C_viols))
        end
    end
    @printf(" : %7.2f\n", maximum(exec_times))

    if save_gif
        println("Saving gif...$gif_name")
        gif(anim, gif_name, fps=10)
    end
end

function get_environment(environment::Symbol; 
    continuous::Bool = true,
    num_cars::Int = 2,
)
    if environment == :cp
        env = CartPoleEnv(continuous=continuous, rng=MersenneTwister())
    elseif environment == :mc
        env = MountainCarEnv(continuous=continuous, rng=MersenneTwister())
    elseif environment == :cr
        env = CarRacingEnv(rng=MersenneTwister())
    elseif environment == :mcr
        env = MultiCarRacingEnv(num_cars, rng=MersenneTwister())
    else
        error("Not implemented for $environment")
    end
    return env
end

function get_policy(
    env, policy_type, num_samples, horizon, λ, 
    α, U₀, cov_mat, opt_its, ce_elite_threshold, 
    step_factor, σ, elite_perc_threshold, pol_log,
    Σ_est, λ_ais,
)
    if policy_type == :gmppi
        pol = GMPPI_Policy(env, 
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
        pol = CEMPPI_Policy(env, 
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=opt_its,
            ce_elite_threshold=ce_elite_threshold,
            Σ_est = Σ_est,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :cmamppi
        pol = CMAMPPI_Policy(env, 
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=opt_its,
            σ=σ,
            elite_perc_threshold=elite_perc_threshold,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :nesmppi
        pol = NESMPPI_Policy(env, 
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=opt_its,
            step_factor=step_factor,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :aismppi
        pol = μΣAISMPPI_Policy(env, 
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=opt_its,
            λ_ais=λ_ais,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :amismppi
        pol = AMISMPPI_Policy(env, 
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            opt_its=opt_its,
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
            opt_its=opt_its,
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
            opt_its=opt_its,
            λ_ais=λ_ais,
            log=pol_log,
            rng=MersenneTwister(),
        )
    elseif policy_type == :mppi
        pol = MPPI_Policy(env, 
            num_samples=num_samples,
            horizon=horizon,
            λ=λ,
            α=α,
            U₀=U₀,
            cov_mat=cov_mat,
            log=pol_log,
            rng=MersenneTwister(),
            )
    else
        error("No policy_type of $policy_type")
    end
    return pol
end

function quantile_ci(x, p=0.05, q=0.5)
    n = length(x)
    zm = quantile(Normal(0.0, 1.0), p/2)
    zp = quantile(Normal(0.0, 1.0), 1-p/2)
    j = Int(ceil(n*q + zm*sqrt(n*q*(1-q))))
    k = Int(ceil(n*q + zp*sqrt(n*q*(1-q))))
    x_sorted = sort(x)
    return x_sorted[j], quantile(x, q), x_sorted[k]
end

for ii = 1:1
    pol_type = :mppi
    
    num_cars = 1
    ii == 1
    ns = 375
    oIts = 1
    λ = 10.0
    λ_ais = 20.0
    
    Σ_est = :mle

    seeded = nothing

    sim_type            = :cr
    num_cars            = num_cars
    n_trials            = 1
    laps                = 2

    p_type              = pol_type
    n_steps             = 25
    n_samp              = ns
    horizon             = 50
    λ                   = λ
    α                   = 1.0
    opt_its             = oIts
    λ_ais               = λ_ais
    ce_elite_threshold  = 0.8
    min_max_sample_p    = 0.0 
    step_factor         = 0.0001
    σ                   = 0.5
    elite_perc_threshold= 0.8
    U₀                  = zeros(Float64, num_cars*2)
    cov_mat             = block_diagm([0.0625, 0.1], num_cars)

    Σ_est_target        = Σ_est

    state_x_sigma       = 0.0
    state_y_sigma       = 0.0
    state_ψ_sigma       = 0.0

    seeded              = seeded

    plot_steps          = false
    pol_log             = false
    plot_traj           = false
    traj_p              = 1.0
    save_gif            = false

    println("Sim Type:              $sim_type")
    println("Num Cars:              $num_cars")
    println("Trials:                $n_trials")
    println("Laps:                  $laps")
    println("Policy Type:           $p_type")
    println("# Samples:             $n_samp")
    println("Horizon:               $horizon")
    println("λ:                     $λ")
    println("α:                     $α")
    println("# Opt Iterations:      $opt_its")
    println("λ_ais:                 $λ_ais")
    println("CE Elite Threshold:    $ce_elite_threshold")
    println("Min Max Sample Perc:   $min_max_sample_p")
    println("NES Step Factor:       $step_factor")
    println("CMA Step Factor (σ):   $σ")
    println("CMA Elite Perc Thres:  $elite_perc_threshold")
    println("U₀:                    zeros(Float64, $(num_cars*2))")
    println("Σ:                     block_diagm([0.0625, 0.1], $num_cars)")
    println("Σ_est       :          $Σ_est")
    println("State X σ:             $state_x_sigma")
    println("State Y σ:             $state_y_sigma")
    println("State ψ σ:             $state_ψ_sigma")
    println("Seeded:                $seeded")
    println()

    simulate_environment(sim_type, 
        num_cars = num_cars,
        num_steps = n_steps, 
        num_trials = n_trials, 
        laps = laps,
        policy_type=p_type, 
        num_samples=n_samp, 
        horizon=horizon,  
        λ = λ,
        α = α,
        U₀ = U₀,
        cov_mat = cov_mat,
        opt_its = opt_its,
        λ_ais=λ_ais,
        ce_elite_threshold = ce_elite_threshold,
        min_max_sample_perc = min_max_sample_p,
        step_factor = step_factor,
        σ = σ,
        elite_perc_threshold = elite_perc_threshold,
        pol_log=pol_log,
        plot_traj=plot_traj,
        plot_traj_perc = traj_p,
        save_gif=save_gif, 
        plot_steps=plot_steps,
        seeded=seeded,
        state_x_sigma=state_x_sigma,
        state_y_sigma=state_y_sigma,
        state_ψ_sigma=state_ψ_sigma,
        Σ_est=Σ_est,
    )
end

# for ii = 2:2
#     oIts = 8
#     λ = 0.1
#     λ_ais = 0.1
#     if ii == 1
#         pol_type = :mppi
#         ns = 20*oIts
#         oIts = 1    
#     elseif ii == 2
#         pol_type = :cemppi
#         ns = 20
#     elseif ii == 3
#         pol_type = :pmcmppi
#         ns = 20
#     elseif ii == 4
#         pol_type = :amismppi
#         ns = 20
#     elseif ii == 5
#         pol_type = :μaismppi
#         ns = 20
#     elseif ii == 6
#         pol_type = :cmamppi
#         ns = 20
#     end
    
#     Σ_est = Σ_est

#     seeded = nothing

#     sim_type            = :mc
#     n_trials            = 25

#     p_type              = pol_type
#     n_steps             = 200
#     n_samp              = ns
#     horizon             = 15
#     λ                   = λ
#     α                   = 1.0
#     opt_its             = oIts
#     λ_ais               = λ_ais
#     ce_elite_threshold  = 0.8 
#     σ                   = 0.5
#     elite_perc_threshold= 0.8
#     U₀                  = [0.0]
#     cov_mat             = [1.5]

#     Σ_est               = Σ_est

#     seeded              = seeded

#     plot_steps          = false
#     pol_log             = false
#     plot_traj           = false
#     traj_p              = 1.0
#     save_gif            = false

#     println("Sim Type:              $sim_type")
#     println("Trials:                $n_trials")
#     println("Policy Type:           $p_type")
#     println("# Samples:             $n_samp")
#     println("Horizon:               $horizon")
#     println("λ:                     $λ")
#     println("α:                     $α")
#     println("# Opt Iterations:      $opt_its")
#     println("λ_ais:                 $λ_ais")
#     println("CE Elite Threshold:    $ce_elite_threshold")
#     println("CMA Step Factor (σ):   $σ")
#     println("CMA Elite Perc Thres:  $elite_perc_threshold")
#     println("U₀:                    [0.0]")
#     println("Σ:                     [1.5]")
#     println("Σ_est       :          $Σ_est")
#     println("Seeded:                $seeded")
#     println()

#     simulate_environment(sim_type, 
#         num_steps = n_steps, 
#         num_trials = n_trials, 
#         policy_type=p_type, 
#         num_samples=n_samp, 
#         horizon=horizon,  
#         λ = λ,
#         α = α,
#         U₀ = U₀,
#         cov_mat = cov_mat,
#         opt_its = opt_its,
#         λ_ais=λ_ais,
#         ce_elite_threshold = ce_elite_threshold,
#         σ = σ,
#         elite_perc_threshold = elite_perc_threshold,
#         pol_log=pol_log,
#         plot_traj=plot_traj,
#         plot_traj_perc = traj_p,
#         save_gif=save_gif, 
#         plot_steps=plot_steps,
#         seeded=seeded,
#         Σ_est=Σ_est,
#     )
# end