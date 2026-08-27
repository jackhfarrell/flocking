# Pure schedule-selection seams for the calibration campaign (issue 08).
#
# Kept dependency-free and include-able so both the cluster/analysis scripts and the
# test suite can pull them in without dragging along CairoMakie or the SDE solvers.

# Production dt(v) from anchor dt-convergence rows. Each row is (; v, dt, gap): the largest
# pointwise |F_coarse(r,t) - F_fine(r,t)| the CRN harness measured for that (v, dt) (see
# dt_convergence_crn.jl). For each v the coarsest dt whose gap is within tol is
# production-adequate (coarser = cheaper); if no candidate converged we fall back to the
# finest dt tried and flag it converged=false.
function select_production_dt(rows; tol=0.005)
    velocities = unique(row.v for row in rows)
    schedule = NamedTuple{(:v, :dt, :gap, :converged),
        Tuple{Float64,Float64,Float64,Bool}}[]
    for v in velocities
        candidates = [row for row in rows if row.v == v]
        converged = [row for row in candidates if row.gap < tol]
        if isempty(converged)
            row = candidates[argmin([c.dt for c in candidates])]
            push!(schedule, (; v, dt=row.dt, gap=row.gap, converged=false))
        else
            row = converged[argmax([c.dt for c in converged])]
            push!(schedule, (; v, dt=row.dt, gap=row.gap, converged=true))
        end
    end
    sort!(schedule; by=s -> s.v)
    return schedule
end

# Production T_max window per anchor from a scan over contiguous lag windows. Each
# candidate is (; window_end, zeta, objective): the best-fit collapse exponent and its
# least-squares objective (reduced chi2) when the fit uses lags up to window_end.
# Candidates must be ordered by increasing window_end. A window is "stable" when zeta
# moves by no more than zeta_tol against every present neighbour (shrink by one, extend
# by one); among stable windows we take the best objective, else the global best.
function select_stable_window(candidates; zeta_tol=0.005)
    isempty(candidates) && throw(ArgumentError("no candidate windows"))
    stable = Int[]
    for k in eachindex(candidates)
        ok = true
        k > firstindex(candidates) &&
            (ok &= abs(candidates[k].zeta - candidates[k - 1].zeta) <= zeta_tol)
        k < lastindex(candidates) &&
            (ok &= abs(candidates[k].zeta - candidates[k + 1].zeta) <= zeta_tol)
        ok && push!(stable, k)
    end
    pool = isempty(stable) ? collect(eachindex(candidates)) : stable
    best = pool[argmin([candidates[k].objective for k in pool])]
    return (; selected=candidates[best], stable=!isempty(stable) && best in stable)
end
