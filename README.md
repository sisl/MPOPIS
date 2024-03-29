# MPOPIS (Model Predictive Optimized Path Integral Strategies)

[Short YouTube video talking about MPOPIS](https://youtu.be/-7jHJP37Nio)

[arXiv Paper](https://arxiv.org/abs/2203.16633)

A version of model predictive path integral control (MPPI) that allows for the implementation of adaptive importance sampling (AIS) algorithms into the original importance sampling step. Model predictive optimized path integral control (MPOPI) is more sample efficient than MPPI achieving better performance with fewer samples. A video of MPPI and MPOPI controlling 3 cars side by side for comparison can be seen [here](https://youtu.be/dDifSfxtuls). More details can be found in the [wiki](../../wiki/MPOPIS-Details).

The addition of AIS enables the algorithm to use a better set of samples for the calculation of the control. A depiction of how the samples evolve over iterations can be seen in the following gif.
#### MPOPI (CE) 150 Samples, 10 Iterations
<img src="https://github.com/sisl/MPOPIS/blob/main/gifs/CE%20150-10%20AIS%20Iteration.gif" width="600" height="337" />

## Policy Options
Versions of MPPI and MPOPI implemented
 - MPPI and GMPPI
   - [MPPI](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L104) (`:mppi`): Model Predictive Path Integral Control[^1][^2]
   - [GMPPI](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L280) (`:gmppi`): generalized version of MPPI, treating the control sequence as one control vector with a combined covariance matrix
 - MPOPI
   - [i-MPPI](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L317) (`:imppi`): iterative version of MPOPI similar to μ-AIS but without the decoupled inverse temperature parameter. μ-AIS is equivalent to IMPPI when λ_ais = λ.
   - [PMC](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L744) (`:pmcmppi`): population Monte Carlo algorithm with one distribution[^3]
   - [μ-AIS](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L608) (`:μaismppi`): mean only moment matching AIS algorithm
   - [μΣ-AIS](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L673) (`:μΣaismppi`): mean and covariance moment matching AIS algorithm similar to Mixture-PMC[^4]
   - [CE](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L375) (`:cemppi`): cross-entropy method[^5][^6]
   - [CMA](https://github.com/sisl/MPOPIS/blob/157f2d8dc94d71e39208eba8c8bc4f8061e31d83/src/mppi_mpopi_policies.jl#L474) (`:cmamppi`): covariance matrix adaptation evolutionary strategy[^5][^7]

**For implementation details reference the source code. For simulation parameters used, reference the [wiki](../../wiki/MPOPIS-Details).**

## Getting Started
Use the julia package manager to add the MPOPIS module:
```julia
] add https://github.com/sisl/MPOPIS
using MPOPIS
```
If you want to use the MuJoCo environments, ensure you have `envpool` installed in your `PyCall` distribution:
```julia
install_mujoco_requirements()
```

Now, we can use the built-in example to simulate the MountainCar environment:
```julia
simulate_mountaincar(policy_type=:cemppi, num_trials=5)
```

Simulate the Car Racing environment and save a gif:
```julia
simulate_car_racing(save_gif=true)
```

<img src="https://github.com/sisl/MPOPIS/blob/main/gifs/cr-1-cemppi-150-50-10.0-1.0-10-0.8-ss-1-2.gif" width="450" height="450" />

Also plotting the trajectories and simulating multiple cars
```julia
simulate_car_racing(num_cars=3, plot_traj=true, save_gif=true)
```
<img src="https://github.com/sisl/MPOPIS/blob/main/gifs/mcr-3-cemppi-150-50-10.0-1.0-10-0.8-ss-1-2.gif" width="450" height="450" />

Run a MuJoCo environment:
```julia
simulate_envpool_env(
    "HalfCheetah-v4";
    frame_skip = 5,
    num_trials = 2,
    policy_type = :cemppi,
    num_steps = 50,
    num_samples = 100,
    ais_its = 5,
    λ = 1.0,
    ce_Σ_est = :ss,
    seed = 1,
    output_acts_file = true,
)
```
The output should be something similar to:
```
Env Name:                     HalfCheetah-v4
Num Trails:                   2
Num Steps:                    50
Policy Type:                  cemppi
Num samples                   100
Horizon                       50
λ (inverse temp):             1.00
α (control cost param):       1.00
# AIS Iterations:             5
CE Elite Threshold:           0.80
CE Σ Est Method:              ss
U₀                            [0.0000, ..., 0.0000]
Σ                             0.2500 0.2500 0.2500 0.2500 0.2500 0.2500 
Seed:                         1

Trial    #:       Reward :   Steps:  Reward/Step : Ex Time
Trial    1:       115.46 :      50:         2.31 :   19.55
Trial    2:       126.08 :      50:         2.52 :   19.53
-----------------------------------
Trials AVE:       120.77 :   50.00:         2.42 :   19.54
Trials STD:         7.51 :    0.00:         0.15 :    0.02
Trials MED:       120.77 :   50.00:         2.42 :   19.54
Trials L95:       115.46 :   50.00:         2.31 :   19.53
Trials U95:       126.08 :   50.00:         2.52 :   19.55
Trials MIN:       115.46 :   50.00:         2.31 :   19.53
Trials MAX:       126.08 :   50.00:         2.52 :   19.55
```

The `output_acts_file` option, outputs a csv with the actions for the given environment. If you have the required python libraries installed (i.e. gym, numpy, imageio, and argparse), you can use the provided python script to generate a gif. By default, the `simulate_envpool_env` function outputs the action csv into the `./acts` directory. The parameters to `make_mujoco_gif.py` are
 - `-env`: environment name (e.g. 'Ant-v4')
 - `-af`: action csv file
 - `-o`: output gif file name without the extension (e.g. 'output_fname')

Using one of the above action files:
```
python ./src/envs/make_mujoco_gif.py -env HalfCheetah-v4 -af ./acts/HalfCheetah-v4_5_cemppi_50_2_1_50_1.0_1.0_0.0_0.25_100_5_0.8_sstrial-2.csv -o HalfCheetah-v4_output_gif
```
<img src="https://github.com/sisl/MPOPIS/blob/main/gifs/HalfCheetah-v4_output_gif.gif" width="450" height="450" />


# Citation
```
@inproceedings{Asmar2023},
title = {Model Predictive Optimized Path Integral Strategies},
author = {Dylan M. Asmar and Rasalu Senanayake and Shawn Manuel and Mykel J. Kochenderfer},
booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
year = {2023}
```

# Questions
### In the paper, during the control cost computation (Algorithm 1, line 9) the noise is sampled from Σ′, but the given Σ is utilized for the inversion, is this a typo?
The control cost is computed using the covariance matrix Σ. As we adjust our distribution, we calculate the control cost based on the original distribution and account for the appropriate change in control amount as we change our proposal distribution.

### The algorithm does not use the updated covariance from one MPC iteration to the next, why is this the case?
Using the updated covariance matrix in subsequent iterations of the MPC algorithm could result in faster convergence and during the next AIS iterations. However, it would likely decrease robustness (without robustness considerations in the AIS step). We considered showing results that used the updated covariance matrix, but wanted to focus on the core contributions of the paper and left that for future work.

### In the algorithm, the trajectory costs utilized to update the control parameters are not centered or normalized, is this intentional? 
This was intentional to align with previous versions of MPPI in the literature. There are other approaches that adjust the costs for numerical stability and to ease tuning across different environments. We do not anticipate a major change in performance if a step to adjust the costs is added. 

### How does this compare to a version where MPPI is allowed to run a few iterations before using the control output? This approach is similar to previous work[^8] and other versions of MPC approaches.
An iterative version of MPPI is similar the approach we take in the paper. The main differences are the decoupling of the inverse temperature parameter and the ability to sample from a joint distribution versus separate distributions at each control step. The performance of μ-AIS is similar to the iterative version and outperformed a pure iterative MPPI version in our experiments.


# References

[^1]: G. Williams, N. Wagener, B. Goldfain, P. Drews, J. M. Rehg, B. Boots, and E. A. Theodorou. Information theoretic MPC for model-based reinforcement learning. Proceedings - IEEE International Conference on Robotics and Automation, 2017.
[^2]: G. R. Williams. Model predictive path integral control: Theoretical foundations and applications to autonomous driving. PhD thesis, Georgia Institute of Technology, 2019.
[^3]: O. Capp´e, A. Guillin, J. M. Marin, and C. P. Robert. Population Monte Carlo. Journal of Computational and Graphical Statistics, 13:907–929, 2004.
[^4]: O. Capp´e, R. Douc, A. Guillin, J. M. Marin, and C. P. Robert. Adaptive importance sampling in general mixture classes. Statistics and Computing, 18, 2008. 
[^5]: M. J. Kochenderfer and T. A. Wheeler. Algorithms for Optimization. MIT Press, 2019.
[^6]: R. Y. Rubinstein and D. P. Kroese. The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning. Vol. 133. New York: Springer, 2004.
[^7]: Y. El-Laham, V. Elvira, and M. F. Bugallo. Robust covariance adaptation in adaptive importance sampling. IEEE Signal Processing Letters, 25, 2018. 
[^8]: J. Pravitra, E. A. Theodorou, and E. N. Johnson, “Flying complex maneuvers with model predictive path integral control,” in AIAA SciTech Forum, 2021.
