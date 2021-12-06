# MPOPIS (Model Predictive Optimized Path Integral Strategies)
A version of of model predictive path integral control (MPPI) that allows for the implementation of adaptive importance sampling (AIS) algorithms into the original importance sampling step. Model predictive optimized path integral control (MPOPI) is more sample efficient than MPPI achieving better performance with fewer samples.

## Getting Started
```julia
] add https://github.com/sisl/MOPOPIS
```

To simulate the MountainCar envrironment and ensure everything is working
```julia
using MPOPIS
simulate_mountaincar(num_trials=5)
```

To simulate the Car Racing environment and save a gif
```julia
simulate_car_racing(save_gif=true)
```

Adding the trajectories and simulating multiple cars
```julia
simulate_car_racing(num_cars=3, plot_traj=true, save_gif=true)
```

## Policy Options
The version of MPPI and MPOPI implemented
 - Model Predictive Path Integral Control (MPPI)
 - Generalized MPPI