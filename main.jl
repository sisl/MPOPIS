

using Profile
using Printf

include("./envs/CarRacing.jl")
include("./envs/MultiCarRacing.jl")
include("./envs/DroneEnv.jl")
include("./agents/mppi.jl")


function RLBase.reward(env::MountainCarEnv{A,T}) where {A,T} 
    rew = 0.0
    
    if env.state[1] >= env.params.goal_pos && 
        env.state[2] >= env.params.goal_velocity
        rew += 100000
    end
    # rew -= (env.state[1] - env.params.goal_pos)^2
    rew += abs(env.state[2])*10
    rew += env.done ? 0.0 : -1.0

    return rew
end

function RLBase.reward(env::CartPoleEnv{A,T}) where {A,T} 
    rew = 0.0

    rew += abs(env.state[3])

    if abs(env.state[1]) > env.params.xthreshold ||
        abs(env.state[3]) > env.params.thetathreshold
        rew += -1000
    end

    rew += env.done ? 0.0 : 1.0
end

function simulate_environment(environment;
    num_trials = 1,
    num_samples = 10, 
    horizon = 15,
    λ = 1.0,
    α = 1.0,
    U₀ = [1.5],
    cov_mat = [1.5],
    policy_type = :gmppi,
    continuous = true,
    plot_steps = false,
    save_gif = false,
    )

    if environment == :cartpole
        env = CartPoleEnv(continuous=continuous)
    elseif environment == :mountaincar
        env = MountainCarEnv(continuous=continuous)
    else
        error("Not implemented for $environment")
    end

    anim = Animation()

    rews = 0
    steps = 0

    @printf("Trials    : %6s : %5s\n", "Reward", "Steps")
    for k ∈ 1:num_trials
        reset!(env)
        
        if policy_type == :gmppi
            pol = GMPPI_Policy(env, 
                num_samples=num_samples,
                horizon=horizon,
                λ=λ,
                α=α,
                U₀=U₀,
                cov_mat=cov_mat,
                )
        elseif policy_type == :mppi
            pol = MPPI_Policy(env, 
                num_samples=num_samples,
                horizon=horizon,
                λ=λ,
                α=α,
                U₀=U₀,
                cov_mat=cov_mat,
                )
        else
            error("No policy_type of $policy_type")
        end
        
        seed!(env, k)
        seed!(pol, k)

        # policy "burn-in"
        # for _ in 1:30
        #     pol(env)
        # end

        rew = 0
        cnt = 0
        while !env.done
            act = pol(env)
            env(act[1])
            cnt += 1
            rew += reward(env)
            if plot_steps || save_gif
                p = plot(env)
                if save_gif frame(anim) end
                if plot_steps display(p) end
            end
        end
        @printf("Trial %4d: %5.1f : %3d\n", k, rew, cnt-1)
        rews += rew
        steps += cnt-1
    end
    
    @printf("Trial %4s: %5.1f : %5.1f\n", "Ave", rews/num_trials, steps/num_trials)

    if save_gif
        println("Saving gif...")
        gif(anim, "temp_gif.gif", fps=10)
    end

end