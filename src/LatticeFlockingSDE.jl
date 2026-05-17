module LatticeFlockingSDE

using DifferentialEquations
using FFTW
using Random
using Statistics
using StochasticDiffEq

export ModelParams, SimulationConfig, RunResult
export site_index, wrap_angles!, xy_energy, compute_mu!, drift!
export random_angles, initial_angles, run_trajectory, run_ensemble
export magnetization, radial_correlation, fit_power_law
export radial_displacement_shells, chunk_correlator, online_mean_stderr!
export expected_eta

include("model.jl")
include("observables.jl")
include("chunked_correlator.jl")
include("simulation.jl")

end
