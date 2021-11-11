using Profile
using Printf
using ProgressMeter
using Dates

include("./envs/CarRacing.jl")
include("./envs/MultiCarRacing.jl")
include("./envs/DroneEnv.jl")
include("./agents/mppi.jl")
include("./envs/plots.jl")


function RLBase.reward(env::MountainCarEnv{A,T}) where {A,T} 
    rew = 0.0
    
    if env.state[1] >= env.params.goal_pos && 
        env.state[2] >= env.params.goal_velocity
        rew += 100000
    end

    # if env.state[1] <= env.params.min_pos+0.01
    #     rew += -5000
    # end
    
    rew += abs(env.state[2])*10
    rew += env.done ? 0.0 : -1.0

    return rew
end

function RLBase.reward(env::CartPoleEnv{A,T}) where {A,T} 
    rew = 0.0

    # rew += abs(env.state[3])

    # if abs(env.state[1]) > env.params.xthreshold ||
    #     abs(env.state[3]) > env.params.thetathreshold
    #     rew += -1000
    # end

    rew += env.done ? 0.0 : 1.0
end


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
    ce_its = 10,
    ce_elite_threshold = 0.8,
    pol_log = false,
    )

    gif_name = "$environment-$num_cars-$policy_type-$num_samples-$horizon-$λ-$α-"
    cov_v = cov_mat[1,1]
    gif_name = gif_name * "$cov_v-"
    if policy_type == :cemppi
        gif_name = gif_name * "$ce_its-$ce_elite_threshold-"
    end
    gif_name = gif_name * "$num_trials-$laps.gif"

    anim = Animation()

    rews = zeros(Float64, num_trials)
    steps = zeros(Float64, num_trials)
    lap_ts = [zeros(Float64, num_trials) for _ in 1:laps]
    mean_vs = zeros(Float64, num_trials)
    max_vs = zeros(Float64, num_trials)
    mean_βs = zeros(Float64, num_trials)
    max_βs = zeros(Float64, num_trials)
    β_viols = zeros(Float64, num_trials)
    T_viols = zeros(Float64, num_trials)
    C_viols = zeros(Float64, num_trials)
    exec_times = zeros(Float64, num_trials)  

    @printf("Trial    #: %12s : %7s", "Reward", "Steps")
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
        pol = get_policy(env, policy_type, num_samples, horizon,
                λ, α, U₀, cov_mat, ce_its, ce_elite_threshold,
                pol_log,
        )
        seed!(env, 36)
        seed!(pol, 36)
        
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
                    within_t = within_track(env)
                    if ex_β β_viol += 1 end
                    if !within_t trk_viol += 1 end
                    temp_rew = step_rew + ex_β*5000 + !within_t*10000
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

        @printf("Trial %4d: %12.2f : %7d", k, rew, cnt-1)
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

    @printf("Trials %3s: %12.2f : %7.2f", "AVE", mean(rews), mean(steps))
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
    
    @printf("Trials %3s: %12.2f : %7.2f", "STD", std(rews), std(steps))
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

function get_policy(env, policy_type,
    num_samples, horizon, λ, α, U₀, cov_mat, 
    ce_its, ce_elite_threshold, pol_log,
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
            ce_its=ce_its,
            ce_elite_threshold=ce_elite_threshold,
            Σ_est_target = DiagonalUnequalVariance(),
            Σ_est_shrinkage = :lw,
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


for ii = 1:1

    pol_type = :gmppi
    ns = 1500
    ceIts = 10

    # if ii == 1
    #     pol_type = :cemppi
    #     ns = 300
    #     ceIts = 5
    # elseif ii == 2
    #     pol_type = :cemppi
    #     ns = 500
    #     ceIts = 3
    # elseif ii == 3
    #     pol_type = :cemppi
    #     ns = 750
    #     ceIts = 2
    # elseif ii == 4
    #     pol_type = :cemppi
    #     ns = 1500
    #     ceIts = 1
    # end

    sim_type            = :cr
    num_cars            = 1
    n_trials            = 1
    laps                = 2

    p_type              = pol_type
    n_steps             = 2000
    n_samp              = ns
    horizon             = 50
    λ                   = 0.5
    α                   = 1.0
    ce_its              = ceIts
    ce_elite_threshold  = 0.8
    U₀                  = zeros(Float64, num_cars*2)
    cov_mat             = block_diagm([0.0625, 0.1], num_cars)

    plot_steps          = false
    pol_log             = true
    plot_traj           = true
    traj_p              = 1.0
    save_gif            = true

    println("Sim Type:              $sim_type")
    println("Num Cars:              $num_cars")
    println("Trials:                $n_trials")
    println("Laps:                  $laps")
    println("Policy Type:           $p_type")
    println("# Samples:             $n_samp")
    println("Horizon:               $horizon")
    println("λ:                     $λ")
    println("α:                     $α")
    println("CE Iterations:         $ce_its")
    println("CE Elite Threshold:    $ce_elite_threshold")
    println("U₀:                    zeros(Float64, $(num_cars*2))")
    println("Σ:                     block_diagm([0.0625, 0.1], $num_cars)")
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
        ce_its = ce_its,
        ce_elite_threshold = ce_elite_threshold,
        pol_log=pol_log,
        plot_traj=plot_traj,
        plot_traj_perc = traj_p,
        save_gif=save_gif, 
        plot_steps=plot_steps,
    )

    # simulate_environment(:mc, 
    #     num_steps=200, 
    #     num_trials=1, 
    #     policy_type=:cemppi, 
    #     num_samples=100, 
    #     horizon=15, 
    #     continuous=true, 
    #     U₀=[0.0], 
    #     pol_log=true,
    #     save_gif=false, 
    #     plot_steps=false
    # )
    # simulate_environment(:cp, 
    #     num_steps=200, 
    #     num_trials=1, 
    #     policy_type=:cemppi, 
    #     num_samples=100, 
    #     horizon=15, 
    #     continuous=true, 
    #     U₀=[0.0], 
    #     pol_log=true,
    #     save_gif=false, 
    #     plot_steps=false
    # )

end