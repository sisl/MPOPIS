# MPOPIS (Model Predictive Optimized Path Integral Strategies)
A version of of model predictive path integral control (MPPI) that allows for the implementation of adaptive importance sampling (AIS) algorithms into the original importance sampling step. Model predictive optimized path integral control (MPOPI) is more sample efficient than MPPI achieving better performance with fewer samples. A video of MPPI and MPOPI controlling 3 cars side by side for comparison can be seen [here](https://youtu.be/dDifSfxtuls).

The addition of AIS enables the algorithm to use a better set of samples for the calculation of the control. A depiction of how the samples evolve over iterations can be seen in the following gif.
#### MPOPI (CE) 150 Samples, 10 Iterations
<img src="https://github.com/sisl/MPOPIS/blob/main/gifs/CE%20150-10%20AIS%20Iteration.gif" width="600" height="337" />


## Policy Options
Versions of MPPI and MPOPI implemented
 - Non-Iterative MPPI and GMPPI
   - [MPPI](https://github.com/sisl/MPOPIS/blob/b89b71102a4a751b56b5aa151751f07527c75c29/src/mppi_mpopi_policies.jl#L99) (`:mppi`): Model Predictive Path Integral Control[^1][^2]
   - [GMPPI](https://github.com/sisl/MPOPIS/blob/b89b71102a4a751b56b5aa151751f07527c75c29/src/mppi_mpopi_policies.jl#L213) (`:gmppi`): generalized version of MPPI, treating the control sequence as one control vector with a combined covariance matrix
 - MPOPI
   - [PMC](https://github.com/sisl/MPOPIS/blob/b89b71102a4a751b56b5aa151751f07527c75c29/src/mppi_mpopi_policies.jl#L626) (`:pmcmppi`): population Monte Carlo algorithm with one distribution[^3]
   - [μ-AIS](https://github.com/sisl/MPOPIS/blob/b89b71102a4a751b56b5aa151751f07527c75c29/src/mppi_mpopi_policies.jl#L490) (`:μaismppi`): mean only moment matching AIS algorithm
   - [μΣ-AIS](https://github.com/sisl/MPOPIS/blob/b89b71102a4a751b56b5aa151751f07527c75c29/src/mppi_mpopi_policies.jl#L555) (`:μΣaismppi`): mean and covariance moment matching AIS algorithm similar to Mixture-PMC[^4]
   - [CE](https://github.com/sisl/MPOPIS/blob/b89b71102a4a751b56b5aa151751f07527c75c29/src/mppi_mpopi_policies.jl#L253) (`:cemppi`): cross-entropy method[^5][^6]
   - [CMA](https://github.com/sisl/MPOPIS/blob/b89b71102a4a751b56b5aa151751f07527c75c29/src/mppi_mpopi_policies.jl#L356) (`:cmamppi`): covariance matrix adaptation evolutionary strategy[^5][^7]

**For implementation details reference the source code. For simulation parameters used, reference the [wiki](../../wiki/MPOPIS-Details).**

## Getting Started
Use the julia package manager to add the MPOPIS module:
```julia
] add https://github.com/sisl/MOPOPIS
```

Using the built in example to simulate the MountainCar envrironment:
```julia
using MPOPIS
simulate_mountaincar(policy_type=:cemppi, num_trials=5)
```

Simulate the Car Racing environment and save a gif:
```julia
simulate_car_racing(save_gif=true)
```

<img src="https://github.com/sisl/MPOPIS/blob/main/gifs/cr-1-cemppi-150-50-10.0-1.0-10-0.8-ss-1-2.gif" width="750" height="750" />

Also plotting the trajectories and simulating multiple cars
```julia
simulate_car_racing(num_cars=3, plot_traj=true, save_gif=true)
```
<img src="https://github.com/sisl/MPOPIS/blob/main/gifs/mcr-3-cemppi-150-50-10.0-1.0-10-0.8-ss-1-2.gif" width="750" height="750" />

[^1]: Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M. Rehg, Byron Boots, and Evangelos A. Theodorou. Information theoretic MPC for model-based reinforcement learning. Proceedings - IEEE International Conference on Robotics and Automation, 2017. doi: 10.1109/ICRA.2017.7989202.
[^2]: Grady Robert Williams. Model predictive path integral control: Theoretical foundations and applications to autonomous driving. PhD thesis, Georgia Institute of Technology, 2019.
[^3]: O Capp´e, A Guillin, JMMarin, and C P Robert. Population Monte Carlo. Journal of Computational and Graphical Statistics, 13:907–929, 2004. doi: 10.1198/106186004X12803.
[^4]: Olivier Capp´e, Randal Douc, Arnaud Guillin, Jean Michel Marin, and Christian P. Robert. Adaptive importance sampling in general mixture classes. Statistics and Computing, 18, 2008. doi: 10.1007/s11222-008-9059-x.
[^5]: Mykel J. Kochenderfer and Tim A. Wheeler. Algorithms for Optimization. MIT Press, 2019.
[^6]: Reuven Y Rubinstein and Dirk P Kroese. The Cross Entropy Method: A Unified Approach To Combinatorial Optimization, Monte-Carlo Simulation (Information Science and Statistics). Springer-Verlag, 2004.
[^7]: Yousef El-Laham, Victor Elvira, and Monica F. Bugallo. Robust covariance adaptation in adaptive importance sampling. IEEE Signal Processing Letters, 25, 2018. doi: 10.1109/LSP.2018.2841641.
