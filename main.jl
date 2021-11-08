

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
    polt_traj_perc = 1.0,
    laps = 2,
    
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

    anim = Animation()

    rews = 0
    steps = 0

    @printf("Trials    : %12s : %6s", "Reward", "Steps")
    if environment == :carracing
        for ii ∈ 1:laps
            @printf(" : %4s%d", "lap ", ii)
        end
        @printf(" : %6s", "BoundV")
    end
    @printf(" : %7s", "Ex Time")
    @printf("\n")
    for k ∈ 1:num_trials
        # Start timer
        time_start = Dates.now()

        env = get_environment(environment, continuous);
        pol = get_policy(env, policy_type, num_samples, horizon,
                λ, α, U₀, cov_mat, ce_its, ce_elite_threshold,
                pol_log,
        )
        seed!(env, k)
        seed!(pol, k)
        
        pm = Progress(num_steps, 1, "Trial $k ....", 50)
        rew, cnt, lap, bound_viol, prev_y = 0, 0, 0, 0, 0
        lap_time = zeros(Int, laps)
        while !env.done && cnt <= num_steps
            act = pol(env)
            env(act)
            cnt += 1
            step_rew = reward(env)
            rew += step_rew
            if plot_steps || save_gif
                if plot_traj && pol_log
                    p = plot(env, pol, polt_traj_perc)
                else 
                    p = plot(env)
                end
                if save_gif frame(anim) end
                if plot_steps display(p) end
            end
            next!(pm)

            if environment == :carracing
                curr_y = env.state[2]
                if step_rew < -8000
                    bound_viol += 1
                end
                if prev_y < 0.0 && 
                    curr_y >= 0.0 && 
                    norm(env.state[1:2]) <= 15.0
                    lap += 1
                    lap_time[lap] = cnt
                end
                if lap >= laps
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
        @printf("Trial %4d: %12.1f : %6d", k, rew, cnt-1)
        if environment == :carracing
            for ii ∈ 1:laps
                @printf(" : %5d", lap_time[ii])
            end
            @printf(" : %6d", bound_viol)
        end
        time_end = Dates.now()
        seconds_ran = Dates.value(time_end - time_start) / 1000
        @printf(" : %7.2f", seconds_ran)
        @printf("\n")

        rews += rew
        steps += cnt-1
    end
    @printf("Trial %4s: %12.1f : %6.1f\n", "Ave", rews/num_trials, steps/num_trials)
    if save_gif
        println("Saving gif...")
        gif(anim, "temp_gif.gif", fps=10)
    end
end

function get_environment(environment::Symbol, continuous::Bool)
    if environment == :cartpole
        env = CartPoleEnv(continuous=continuous)
    elseif environment == :mountaincar
        env = MountainCarEnv(continuous=continuous)
    elseif environment == :carracing
        env = CarRacingEnv()
        env.state[1] = 0.0
        env.state[2] = 0.0
        env.state[3] = deg2rad(90)
        env.state[4] = 10.0
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
            )
    else
        error("No policy_type of $policy_type")
    end
    return pol
end

run_num = 1

if run_num == 1
    simulate_environment(:carracing, 
        num_steps=1350, 
        num_trials=2, 
        laps = 2,

        policy_type=:gmppi, 
        num_samples=500, 
        horizon=50,  
        
        λ = 1.0,
        α = 1.0,
        U₀ = [0.0, 0.0],
        cov_mat = [1.5 0.0 ; 
                0.0 1.5],
        
        ce_its = 10,
        ce_elite_threshold = 0.8,

        pol_log=true,
        plot_traj=true,
        polt_traj_perc = 0.5,

        save_gif=true, 
        plot_steps=false,
    )
elseif run_num == 2
    simulate_environment(:mountaincar, 
        num_steps=200, 
        num_trials=1, 
        policy_type=:cemppi, 
        num_samples=100, 
        horizon=15, 
        continuous=true, 
        U₀=[0.0], 

        pol_log=true,

        save_gif=false, 
        plot_steps=false
    )
elseif run_num == 3
    simulate_environment(:cartpole, 
        num_steps=200, 
        num_trials=1, 
        policy_type=:cemppi, 
        num_samples=100, 
        horizon=15, 
        continuous=true, 
        U₀=[0.0], 

        pol_log=true,

        save_gif=false, 
        plot_steps=false
    )
end