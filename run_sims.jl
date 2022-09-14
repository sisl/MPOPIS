using MPOPIS

simulate_envpool_env(
    "Swimmer-v4";
    frame_skip = 4,
    num_trials = 20,
    policy_type = :mppi,
    num_steps = 100,
    num_samples = 3000,
    λ = 0.1,
    seed = 1,
)

simulate_envpool_env(
    "Swimmer-v4";
    frame_skip = 4,
    num_trials = 20,
    policy_type = :cemppi,
    num_steps = 100,
    num_samples = 600,
    ais_its = 5,
    λ = 0.1,
    ce_Σ_est = :ss,
    seed = 1,
)

# simulate_envpool_env(
#     "Ant-v4";
#     frame_skip = 5,
#     num_trials = 5,
#     policy_type = :mppi,
#     num_steps = 100,
#     num_samples = 3000,
#     λ = 1.0,
#     seed = 1,
# )

# simulate_envpool_env(
#     "Ant-v4";
#     frame_skip = 5,
#     num_trials = 5,
#     policy_type = :cemppi,
#     num_steps = 100,
#     num_samples = 375,
#     λ = 1.0,
#     ais_its = 8,
#     ce_Σ_est = :ss,
#     seed = 1,
# )
