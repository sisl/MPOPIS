using MPOPIS

# simulate_envpool_env(
#     "Swimmer-v4";
#     num_trials=5,
#     policy_type=:mppi,
#     num_steps=50,
#     num_samples=1500
# )

simulate_envpool_env(
    "Swimmer-v4";
    num_trials=5,
    policy_type=:cemppi,
    num_steps=50,
    num_samples=300,
    ais_its=5,
    ce_Σ_est = :mle,
)
