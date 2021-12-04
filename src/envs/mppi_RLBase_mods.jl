

function (env::CartPoleEnv{<:Base.OneTo{Int}})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end
function (env::CartPoleEnv{<:ClosedInterval})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end
function (env::MountainCarEnv{<:ClosedInterval})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end
function (env::MountainCarEnv{<:Base.OneTo{Int}})(a::Vector)
    length(a) == 1 || error("Only implented for 1 step")
    env(a[1])
end

function RLBase.reward(env::MountainCarEnv{A,T}) where {A,T} 
    rew = 0.0
    
    if env.state[1] >= env.params.goal_pos && 
        env.state[2] >= env.params.goal_velocity
        rew += 100000
    end
    
    rew += abs(env.state[2])
    rew += env.done ? 0.0 : -1.0

    return rew
end