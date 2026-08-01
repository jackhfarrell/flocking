#!/usr/bin/env julia
# Local plumbing check for the radii cap + geometric lag spacing (issue 02).
# Confirms the window-builder schedule emits a symmetric geometric `times` vector
# and that the collapse seam (`scan_grid`) consumes that non-uniform `times`
# unchanged, recovering injected exponents for both the geometric schedule and the
# degenerate uniform schedule (which reproduces the prior uniform-lag collapse).

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE
include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

dt = 0.001
ntimes = 8
lag_steps = 2000          # T-max / ntimes = 2.0 in time units

uniform = lag_step_schedule(ntimes, lag_steps; spacing=:uniform)
geom = lag_step_schedule(ntimes, lag_steps; spacing=:geometric)

println("uniform times    = ", uniform.cum_steps .* dt)
println("geometric times  = ", geom.cum_steps .* dt)
println("geometric gaps   = ", geom.gaps)
println("advance_gaps      = ", geom.advance_gaps)
@assert geom.advance_gaps == [reverse(geom.gaps); geom.gaps] "advance gaps must be symmetric"
@assert !all(==(geom.gaps[1]), geom.gaps) "geometric gaps must be non-uniform"

# Inject a known scaling form F(r,t) = t^-eta0 * g(r / t^zeta0) with a trough.
eta0 = 0.3
zeta0 = 0.7
g(u) = (u^2 - 1.0) * exp(-u)
radii = collect(0.5:0.5:40.0)

function synthetic_fields(times)
    F_mean = zeros(length(radii), length(times))
    for (ti, t) in enumerate(times)
        t == 0 && continue
        F_mean[:, ti] .= t^(-eta0) .* g.(radii ./ t^zeta0)
    end
    return F_mean, fill(1e-3, size(F_mean))
end

radius_mask = trues(length(radii))
eta_values = collect(0.0:0.05:0.6)
zeta_values = collect(0.4:0.05:1.0)

for (name, schedule) in (("uniform", uniform), ("geometric", geom))
    times = schedule.cum_steps .* dt
    F_mean, F_stderr = synthetic_fields(times)
    time_indices = collect(2:length(times))   # skip the zero lag
    _, best = scan_grid(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        eta_values, zeta_values, 4, 12)
    println("$name schedule: recovered eta = $(best.eta), zeta = $(best.zeta) ",
        "(injected eta = $eta0, zeta = $zeta0)")
    @assert isapprox(best.eta, eta0; atol=0.05)
    @assert isapprox(best.zeta, zeta0; atol=0.05)
end

println("verify_geometric_lags: OK")
