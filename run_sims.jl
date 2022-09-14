using MPOPIS

# simulate_envpool_env(
#     "Swimmer-v4";
#     num_trials = 5,
#     policy_type = :mppi,
#     num_steps = 50,
#     num_samples = 1500,
#     λ = 0.1,
# )

# simulate_envpool_env(
#     "Swimmer-v4";
#     num_trials = 5,
#     policy_type = :cemppi,
#     num_steps = 50,
#     num_samples = 300,
#     λ = 0.1,
#     ais_its = 5,
#     ce_Σ_est = :ss,
# )

simulate_envpool_env(
    "Ant-v4";
    num_trials = 5,
    policy_type = :mppi,
    num_steps = 20,
    num_samples = 1500,
    λ = 1.0,
)

# simulate_envpool_env(
#     "Ant-v4";
#     num_trials = 5,
#     policy_type = :cemppi,
#     num_steps = 20,
#     num_samples = 300,
#     λ = 1.0,
#     ais_its = 5,
#     ce_Σ_est = :ss,
# )
