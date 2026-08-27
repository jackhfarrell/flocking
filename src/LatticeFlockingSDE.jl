module LatticeFlockingSDE

using DifferentialEquations
using FFTW
using Random
using Statistics
using StochasticDiffEq

export ModelParams, SimulationConfig, RunResult
export site_index, wrap_angles!, xy_energy, compute_mu!, drift!
export random_angles, initial_angles, seed_upward_bump!, positive_sin_marker_x
export run_trajectory, run_ensemble
export magnetization, radial_correlation, fit_power_law
export radial_displacement_shells, chunk_correlator, lag_step_schedule, online_mean_stderr!
export observable_window_range, eta_window_range, eta_equilibrium_reached
export equilibrium_window_blocks, equilibrium_stationarity_reached
export expected_eta
export best_collapse, fit_window_robustness

include("model.jl")
include("observables.jl")
include("chunked_correlator.jl")
include("simulation.jl")
include("exponent.jl")
include("collapse.jl")

end
